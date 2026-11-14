"""Phase 4 action/state integration with no network, camera, serial, or audio."""

from dataclasses import dataclass
import json
import socket
import threading
import time as real_time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.ai.scene_description import SceneDescriptionError, SceneDescriptionResult
from vidi.core import phase4
from vidi.core.configuration import (
    CameraSettings, CloudSettings, ModelSettings, SimulationSettings, SpeechSettings,
)
from vidi.core.types import SpeechResult


@pytest.fixture(autouse=True)
def isolated_boundaries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Phase 4 tests must not open a service, physical device, or audio engine")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(phase4, "GeminiSceneDescriber", forbidden)
    monkeypatch.setattr(phase4, "LocalSpeech", forbidden)
    monkeypatch.setattr(phase4, "_open_proximity", forbidden)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", Mock())
    monkeypatch.setattr(phase4, "time", SimpleNamespace(
        monotonic=real_time.monotonic, monotonic_ns=real_time.monotonic_ns,
        sleep=lambda seconds: real_time.sleep(min(seconds, 0.001)),
    ))


def settings(tmp_path, *, output=False, max_words=60):
    return SimpleNamespace(
        camera=CameraSettings(source="sample", width=160, height=120, fps=30),
        simulation=SimulationSettings(preview=True, report_path=tmp_path / "phase4.json",
                                      output_path=tmp_path / "snapshot.png" if output else None),
        models=ModelSettings(), cloud=CloudSettings(), speech=SpeechSettings(max_words=max_words),
        proximity=SimpleNamespace(nearby_mm=1000, stale_after_s=1.0),
    )


class Camera:
    def __init__(self):
        self.count = 0
        self.closed = False
        self.buffer = np.zeros((120, 160, 3), np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self):
        self.count += 1
        self.buffer[:] = self.count % 200
        return self.buffer


class ImmediateTask:
    """Deterministic worker completion; each result is consumed next frame."""

    def __init__(self, operation):
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.failed = False
        try:
            self.value = operation(self.cancel)
        except BaseException:
            self.failed = True
            self.value = None
        self.done.set()

    def close(self):
        self.cancel.set()


@dataclass(frozen=True)
class Snapshot:
    left: str = "unknown"
    right: str = "ok"


class Monitor:
    def __init__(self):
        self.closed = False
        self.fresh = Mock(return_value=Snapshot())
        self.reconnect = Mock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def latest(self):
        return Snapshot()


def preview_keys(monkeypatch, keys):
    iterator = iter(keys)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(iterator))


def install_description(monkeypatch, callback=None):
    describe = Mock(side_effect=callback or (lambda image, question: SceneDescriptionResult(
        "A chair is beside a desk.", "test-model", 0.2)))
    monkeypatch.setattr(phase4, "GeminiSceneDescriber", Mock(return_value=SimpleNamespace(describe=describe)))
    return describe


def install_speech(monkeypatch, callback=None):
    speak = Mock(side_effect=callback or (lambda text, **kwargs: SpeechResult(
        status="completed", text=text, mode="playback", backend="fake")))
    monkeypatch.setattr(phase4, "LocalSpeech", Mock(return_value=SimpleNamespace(speak=speak)))
    return speak


def install_monitor(monkeypatch):
    monitor = Monitor()
    monkeypatch.setattr(phase4, "_open_proximity", lambda settings: monitor)
    monkeypatch.setattr(phase4, "_format_proximity", lambda snapshot, settings:
                        "Near-field left unknown. Near-field right: 50 centimeters, nearby.")
    monkeypatch.setattr(phase4, "_refresh_proximity", lambda snapshot, settings: snapshot)
    return monitor


def test_no_action_never_constructs_cloud_or_speech(monkeypatch, tmp_path):
    camera = Camera()
    preview_keys(monkeypatch, [-1, -1, ord("q")])

    result = phase4.run_phase4_live(settings(tmp_path), proximity=False, speak=True,
                                   open_source=lambda settings: camera)

    assert result.frames_processed == 3 and camera.closed
    assert result.description_status == "not_requested"
    report = json.loads(result.report_path.read_text())
    assert report["schema_version"] == 3 and report["action_totals"]["cloud_requests"] == 0
    assert report["speech"]["status"] == "not_requested"
    cv2.destroyAllWindows.assert_called_once()


def test_d_owns_exact_image_one_request_and_checkpoints_before_speech(monkeypatch, tmp_path):
    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    camera = Camera()
    config = settings(tmp_path, output=True)
    monitor = install_monitor(monkeypatch)

    def describe(image, question):
        report = json.loads(config.simulation.report_path.read_text())
        assert report["description"]["status"] == "pending"
        assert image[80, 100, 0] == 1 and not np.shares_memory(image, camera.buffer)
        assert question == "What is in front of me?"
        return SceneDescriptionResult("A chair is beside a desk.", "test-model", 0.2)

    model = install_description(monkeypatch, describe)

    def speech(text, *, cancel_event):
        report = json.loads(config.simulation.report_path.read_text())
        assert report["speech"]["status"] == "pending"
        assert report["proximity"]["left"] == "unknown"
        assert text.startswith("Near-field left unknown.")
        assert "Camera snapshot from" in text and len(text.split()) <= 60
        assert not cancel_event.is_set()
        return SpeechResult(status="completed", text=text)

    speaker = install_speech(monkeypatch, speech)
    preview_keys(monkeypatch, [ord("d"), -1, ord("d"), ord("q")])

    result = phase4.run_phase4_live(config, proximity=True, speak=True, open_source=lambda settings: camera)

    model.assert_called_once()
    speaker.assert_called_once()
    monitor.fresh.assert_called_once()
    assert monitor.fresh.call_args.kwargs["after_ns"] > 0
    assert monitor.closed and camera.closed
    assert result.description == "A chair is beside a desk."
    assert result.speech.status == "completed"
    report = json.loads(result.report_path.read_text())
    assert report["description"]["frame_id"] == report["output_frame_id"] == 1
    assert report["last_preview_frame_id"] == 4
    assert int(cv2.imread(str(result.output_path))[80, 100, 0]) == 1


def test_cloud_failure_pauses_d_but_p_and_r_remain_local(monkeypatch, tmp_path):
    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    model = install_description(monkeypatch, lambda *args: (_ for _ in ()).throw(SceneDescriptionError("Gemini unavailable.")))
    monitor = install_monitor(monkeypatch)
    camera = Camera()
    preview_keys(monkeypatch, [ord("d"), ord("d"), ord("p"), -1, ord("r"), ord("q")])

    result = phase4.run_phase4_live(settings(tmp_path), proximity=True, speak=False,
                                   open_source=lambda settings: camera)

    model.assert_called_once()
    assert monitor.fresh.call_count == 2  # Failed D feedback, followed by an independent P action.
    monitor.reconnect.assert_called_once()
    report = json.loads(result.report_path.read_text())
    assert result.description_status == "unavailable"
    assert report["cloud_uploads_paused"] is False
    assert report["action_totals"]["local_reads"] == 1
    assert report["action_totals"]["ignored_actions"] == 1


def test_p_while_cloud_pending_speaks_local_once_and_supersedes_d_narration(monkeypatch, tmp_path):
    started, release, local_spoken = threading.Event(), threading.Event(), threading.Event()
    camera = Camera()
    monitor = install_monitor(monkeypatch)

    def describe(image, question):
        started.set()
        assert release.wait(1.0)
        return SceneDescriptionResult("A desk.", "test-model", 0.2)

    model = install_description(monkeypatch, describe)

    def speech(text, **kwargs):
        assert "Camera snapshot" not in text
        local_spoken.set()
        release.set()
        return SpeechResult(status="completed", text=text)

    speaker = install_speech(monkeypatch, speech)

    def key(delay):
        if camera.count == 1:
            return ord("d")
        if camera.count == 2:
            assert started.wait(0.3)
            return ord("p")
        if local_spoken.is_set() and model.call_count == 1 and camera.count >= 8:
            return ord("q")
        assert camera.count < 100, "Workers failed to complete"
        return -1

    monkeypatch.setattr(cv2, "waitKey", key)
    try:
        result = phase4.run_phase4_live(settings(tmp_path), proximity=True, speak=True,
                                       open_source=lambda settings: camera)
    finally:
        release.set()

    assert result.description_status == "ready" and result.description == "A desk."
    assert camera.count >= 8 and camera.closed and monitor.closed
    speaker.assert_called_once()
    monitor.fresh.assert_called_once()


def test_stop_cancels_audio_and_prevents_queued_narration(monkeypatch, tmp_path):
    entered, cancelled = threading.Event(), threading.Event()
    install_description(monkeypatch)

    def speech(text, *, cancel_event):
        entered.set()
        assert cancel_event.wait(1.0)
        cancelled.set()
        return SpeechResult(status="cancelled", text=text)

    speaker = install_speech(monkeypatch, speech)
    camera = Camera()

    def key(delay):
        if camera.count == 1:
            return ord("d")
        if entered.is_set() and not cancelled.is_set():
            return ord("s")
        if cancelled.is_set() and camera.count > 5:
            return ord("q")
        assert camera.count < 100
        return -1

    monkeypatch.setattr(cv2, "waitKey", key)
    result = phase4.run_phase4_live(settings(tmp_path), proximity=False, speak=True,
                                   open_source=lambda settings: camera)

    assert cancelled.is_set() and camera.closed
    assert result.speech.status == "cancelled"
    speaker.assert_called_once()


def test_ctrl_c_final_report_and_bounded_action_history(monkeypatch, tmp_path):
    camera = Camera()

    def key(delay):
        if camera.count == 125:
            raise KeyboardInterrupt
        return ord("s")

    monkeypatch.setattr(cv2, "waitKey", key)
    result = phase4.run_phase4_live(settings(tmp_path), proximity=False, speak=False,
                                   open_source=lambda settings: camera)

    assert result.interrupted and result.frames_processed == 125 and camera.closed
    report = json.loads(result.report_path.read_text())
    assert report["interrupted"] is True
    assert len(report["actions"]) == report["action_records_limit"] == 120
    assert report["action_records_truncated"] is True
    assert report["action_records_dropped"] == 4


def test_sensor_failure_preserves_visual_speech(monkeypatch, tmp_path):
    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    install_description(monkeypatch)
    monitor = install_monitor(monkeypatch)
    monitor.fresh.side_effect = OSError("private serial details")
    speaker = install_speech(monkeypatch)
    camera = Camera()
    preview_keys(monkeypatch, [ord("d"), -1, -1, ord("q")])

    result = phase4.run_phase4_live(settings(tmp_path), proximity=True, speak=True,
                                   open_source=lambda settings: camera)

    utterance = speaker.call_args.args[0]
    assert "left and right are unknown" in utterance
    assert "A chair is beside a desk." in utterance
    assert "private serial details" not in result.report_path.read_text()


def test_feedback_limits_full_utterance_and_preserves_sensor_unknown():
    sensor = "Near-field left unknown. Near-field right: 50 centimeters, nearby."
    result = phase4.compose_feedback("A " + "chair " * 100, sensor, 3.2, 25)
    assert result.startswith(sensor) and "Camera snapshot from 3.2 seconds ago:" in result
    assert len(result.split()) <= 25 and result.endswith("…")
    assert phase4.compose_feedback("A chair.", "Too many sensor words here", 0, 2) == "Sensors unknown."
    assert phase4.compose_feedback(None, sensor, 0, 60) == sensor
    with pytest.raises(ValueError):
        phase4.compose_feedback("A chair.", sensor, float("nan"), 60)


def test_preview_keeps_live_and_snapshot_pixels_separate_and_owned(monkeypatch):
    from vidi.core.types import FramePacket

    live = FramePacket(8, real_time.monotonic_ns(), np.full((240, 320, 3), 40, np.uint8))
    frozen = FramePacket(1, real_time.monotonic_ns(), np.full((240, 320, 3), 120, np.uint8))
    display = phase4.render_phase4_preview(
        live, frozen, {"status": "ready", "text": "A chair."}, proximity_text="Left unknown. Right unknown.",
        cloud_busy=False, audio_busy=False, paused=False,
    )
    np.testing.assert_array_equal(display[200, 100], (40, 40, 40))
    np.testing.assert_array_equal(display[200, 420], (120, 120, 120))
    assert np.all(live.image_bgr == 40) and np.all(frozen.image_bgr == 120)


def test_daemon_task_close_is_bounded_for_hung_work():
    entered, release = threading.Event(), threading.Event()

    def work(cancel):
        entered.set()
        release.wait(2.0)

    task = phase4._Task(work)
    assert entered.wait(0.3)
    started = real_time.monotonic()
    try:
        task.close()
        assert real_time.monotonic() - started < 0.9
        assert task.cancel.is_set() and task.thread.daemon
    finally:
        release.set()


def test_exact_post_action_sensor_pair_is_reaged_before_speech(tmp_path):
    from vidi.sensors.proximity import ProximityObservation, ProximitySnapshot

    now = real_time.monotonic_ns()
    snapshot = ProximitySnapshot(*(
        ProximityObservation(channel, "ok", distance_mm=400,
                             request_sent_monotonic_ns=now - 2_000_000_000,
                             received_monotonic_ns=now - 1_900_000_000)
        for channel in ("left", "right")
    ))
    refreshed = phase4._refresh_proximity(snapshot, settings(tmp_path))
    assert refreshed.left.status == refreshed.right.status == "stale"
    assert refreshed.left.distance_mm is refreshed.right.distance_mm is None
    assert "nearby return" not in phase4._format_proximity(refreshed, settings(tmp_path))


def test_live_pending_report_delay_reages_exact_pair_in_speech_worker(monkeypatch, tmp_path):
    from vidi.core import controller
    from vidi.sensors import proximity as sensors

    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    now = [10.0]
    monkeypatch.setattr(phase4, "time", SimpleNamespace(
        monotonic=lambda: now[0], monotonic_ns=lambda: int(now[0] * 1e9), sleep=lambda seconds: None,
    ))
    monkeypatch.setattr(sensors, "time", SimpleNamespace(monotonic_ns=lambda: int(now[0] * 1e9)))
    snapshot = sensors.ProximitySnapshot(*(
        sensors.ProximityObservation(channel, "ok", distance_mm=400,
                                     request_sent_monotonic_ns=10_000_000_000,
                                     received_monotonic_ns=10_000_000_000)
        for channel in ("left", "right")
    ))
    monitor = Monitor()
    monitor.fresh.return_value = snapshot
    monitor.latest = lambda: sensors.refresh_snapshot(snapshot, 1.0)
    monkeypatch.setattr(phase4, "_open_proximity", lambda settings: monitor)
    install_description(monkeypatch)
    speaker = install_speech(monkeypatch)
    save = controller._save_report
    calls = 0

    def delayed(path, report):
        nonlocal calls
        calls += 1
        save(path, report)
        if calls == 4:  # Pending speech intent is durable before its worker starts.
            now[0] += 1.1

    monkeypatch.setattr(controller, "_save_report", delayed)
    preview_keys(monkeypatch, [ord("d"), -1, -1, ord("q")])
    camera = Camera()

    result = phase4.run_phase4_live(settings(tmp_path), proximity=True, speak=True,
                                   open_source=lambda settings: camera)

    assert "nearby return" not in speaker.call_args.args[0]
    assert "reading unavailable" in speaker.call_args.args[0]
    assert "1.1 seconds ago" in speaker.call_args.args[0]
    report = json.loads(result.report_path.read_text())
    assert report["proximity"]["left"]["status"] == "stale"
    assert report["proximity"]["left"]["distance_mm"] is None
    assert report["speech"]["text"] == speaker.call_args.args[0]
    assert report["capture_fps"] > 0 and report["processing_s"] == pytest.approx(1.1)
    assert report["startup_s"] == 0.0


def test_failed_d_still_speaks_fresh_local_feedback_once(monkeypatch, tmp_path):
    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    install_description(monkeypatch, lambda *args: (_ for _ in ()).throw(SceneDescriptionError("Gemini unavailable.")))
    monitor = install_monitor(monkeypatch)
    speaker = install_speech(monkeypatch)
    preview_keys(monkeypatch, [ord("d"), -1, -1, ord("q")])
    camera = Camera()

    result = phase4.run_phase4_live(settings(tmp_path), proximity=True, speak=True,
                                   open_source=lambda settings: camera)

    speaker.assert_called_once()
    monitor.fresh.assert_called_once()
    assert "Scene description unavailable." in speaker.call_args.args[0]
    assert "Camera snapshot" not in speaker.call_args.args[0]
    assert result.description_status == "unavailable" and result.speech.status == "completed"
    assert json.loads(result.report_path.read_text())["cloud_uploads_paused"] is True


def test_local_p_logs_requested_text_with_speech_off(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    monitor = install_monitor(monkeypatch)
    preview_keys(monkeypatch, [ord("p"), -1, ord("q")])
    caplog.set_level("INFO", logger=phase4.__name__)

    result = phase4.run_phase4_live(settings(tmp_path), proximity=True, speak=False,
                                   open_source=lambda settings: Camera())

    monitor.fresh.assert_called_once()
    assert "Requested feedback: Near-field left unknown." in caplog.text
    assert result.speech.status == "not_requested"


def test_s_cancels_cloud_job_before_it_can_construct_client(monkeypatch, tmp_path):
    jobs = []

    class DeferredTask:
        def __init__(self, operation):
            self.operation = operation
            self.cancel = threading.Event()
            self.done = threading.Event()
            self.value = None
            self.failed = False
            jobs.append(self)

        def close(self):
            self.cancel.set()

    monkeypatch.setattr(phase4, "_Task", DeferredTask)
    camera = Camera()

    def key(delay):
        if camera.count == 1:
            return ord("d")
        if camera.count == 2:
            return ord("s")
        if camera.count == 3:
            assert jobs[0].cancel.is_set()
            jobs[0].value = jobs[0].operation(jobs[0].cancel)
            jobs[0].done.set()
            return -1
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", key)
    result = phase4.run_phase4_live(settings(tmp_path), proximity=False, speak=True,
                                   open_source=lambda settings: camera)

    assert result.description_status == "cancelled"
    assert result.speech.status == "not_requested"
    assert json.loads(result.report_path.read_text())["cloud_uploads_paused"] is False


def test_quit_preserves_finished_speech_worker_outcome_before_next_ui_poll(monkeypatch, tmp_path):
    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    install_description(monkeypatch)
    install_speech(monkeypatch, lambda text, **kwargs: SpeechResult(
        status="completed", text="Actual completed utterance.", mode="playback"))
    preview_keys(monkeypatch, [ord("d"), ord("q")])

    result = phase4.run_phase4_live(settings(tmp_path), proximity=False, speak=True,
                                   open_source=lambda settings: Camera())

    assert result.speech.status == "completed" and result.speech.text == "Actual completed utterance."
    assert json.loads(result.report_path.read_text())["speech"]["text"] == "Actual completed utterance."
