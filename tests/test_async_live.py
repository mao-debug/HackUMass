"""Async live UI and reporting contracts with fake capture/inference only."""

from dataclasses import replace
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

from vidi.core import live
from vidi.core.configuration import ConfigurationError, SimulationSettings, load_settings
from vidi.core.types import Detection, DetectionResult, FramePacket


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_live_boundaries(monkeypatch):
    """A refactor must not turn these tests into a live request or GUI run."""
    def forbidden(*args, **kwargs):
        pytest.fail("Async integration tests must use fake workers, cameras, and displays")

    monkeypatch.setattr(live, "LiveDetector", forbidden)
    monkeypatch.setattr(cv2, "imshow", forbidden)
    monkeypatch.setattr(cv2, "waitKey", forbidden)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(live, "time", SimpleNamespace(
        monotonic=lambda: 8.0, monotonic_ns=lambda: 8_000_000_000, sleep=lambda seconds: None,
    ))


def packet(frame_id, value=None):
    color = value if value is not None else frame_id % 200 + 20
    return FramePacket(
        frame_id, 1_000_000_000 + frame_id * 100_000_000,
        np.full((120, 160, 3), color, np.uint8),
    )


def ready(frame):
    return DetectionResult(
        frame.frame_id, "gemini", "ready",
        (Detection(None, "chair", None, (15, 40, 45, 90)),), model="offline-test-model",
    )


class FakeWorker:
    """Publish only scripted responses; no detector or service is invoked."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.submissions = []
        self.stopped = False
        self.stop_on_unavailable = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = self.stopped = True

    def poll(self):
        result = next(self.responses, None)
        if result is not None and self.stop_on_unavailable and result[1].status != "ready":
            self.stopped = True
        return result

    def submit(self, frame):
        assert not self.stopped
        self.submissions.append(frame.frame_id)


def install_worker(monkeypatch, worker):
    def factory(detector, *, min_interval_s, stop_on_unavailable):
        assert detector is not None
        assert min_interval_s == 5.0 and stop_on_unavailable is True
        worker.stop_on_unavailable = stop_on_unavailable
        return worker

    constructor = Mock(side_effect=factory)
    monkeypatch.setattr(live, "LiveDetector", constructor)
    return constructor


def install_capture(frames):
    capture = MagicMock()
    capture.read_packet.side_effect = frames
    manager = MagicMock()
    manager.__enter__.return_value = capture
    return capture, manager, Mock(return_value=manager)


def settings_for(tmp_path, *, preview=True, output=True):
    settings = load_settings(ROOT / "config/pi-live.yaml")
    return replace(settings, simulation=SimulationSettings(
        preview=preview, async_detection=True,
        output_path=tmp_path / "analyzed.png" if output else None,
        report_path=tmp_path / "live.json",
    ))


def install_preview(monkeypatch, frame_count):
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda title, image: shown.append(image.copy()))
    monkeypatch.setattr(cv2, "waitKey", Mock(side_effect=[-1] * (frame_count - 1) + [ord("q")]))
    return shown


def test_live_pixels_advance_while_analysis_is_pending_and_export_matches_analyzed_frame(monkeypatch, tmp_path):
    frames = [packet(index, index * 30) for index in range(1, 5)]
    originals = [frame.image_bgr.copy() for frame in frames]
    worker = FakeWorker([None, None, (frames[0], ready(frames[0])), None])
    factory = install_worker(monkeypatch, worker)
    capture, manager, open_source = install_capture(frames)

    def release_camera(*args):
        assert worker.closed, "New inference must stop before camera teardown begins"
        return False

    manager.__exit__.side_effect = release_camera
    shown = install_preview(monkeypatch, len(frames))
    settings = settings_for(tmp_path)

    result = live.run_live_detection(settings, object(), open_source=open_source, gemini_live=True)

    assert [int(image[60, 100, 0]) for image in shown] == [30, 60, 90, 120]
    assert [int(image[85, 260, 0]) for image in shown] == [0, 0, 30, 30]
    assert capture.read_packet.call_count == 4 and worker.submissions == [1, 2, 3, 4]
    assert result.frames_processed == 4 and result.ready_frames == 1
    assert result.last_inference.frame_id == 1 and worker.closed
    factory.assert_called_once()
    open_source.assert_called_once_with(settings)
    manager.__exit__.assert_called_once()
    for frame, original in zip(frames, originals):
        np.testing.assert_array_equal(frame.image_bgr, original)
    exported = cv2.imread(str(result.output_path))
    assert int(exported[60, 100, 0]) == 30
    report = json.loads(result.report_path.read_text())
    assert report["output_frame_id"] == report["last_analyzed_frame_id"] == 1
    assert report["last_preview_frame_id"] == 4
    assert report["last_inference"]["frame_id"] == report["frames"][-1]["frame_id"] == 1
    assert report["detections"][0]["class_id"] is None
    assert report["detections"][0]["confidence"] is None


def test_two_panes_use_current_raw_and_original_analyzed_pixels_with_age_and_no_mutation(monkeypatch):
    current, old = packet(42, 35), packet(7, 125)
    old = replace(old, captured_monotonic_ns=6_000_000_000)
    current_original, old_original = current.image_bgr.copy(), old.image_bgr.copy()
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", put_text)

    canvas = live.render_live_preview(current, (old, ready(old)), paused=True)

    assert canvas.shape == (120, 320, 3)
    assert not np.shares_memory(canvas, current.image_bgr)
    assert not np.shares_memory(canvas, old.image_bgr)
    np.testing.assert_array_equal(current.image_bgr, current_original)
    np.testing.assert_array_equal(old.image_bgr, old_original)
    np.testing.assert_array_equal(canvas[60, 100], (35, 35, 35))
    np.testing.assert_array_equal(canvas[60, 260], (125, 125, 125))
    np.testing.assert_array_equal(canvas[40, 15], (35, 35, 35))
    np.testing.assert_array_equal(canvas[40, 175], (60, 220, 80))
    texts = [call.args[1] for call in put_text.call_args_list]
    assert "LIVE CAMERA - frame 42" in texts
    assert "GEMINI SNAPSHOT - PROTOTYPE" in texts
    assert "ANALYZED frame 7 - 2.0s old - UPLOADS PAUSED" in texts


def test_gemini_unavailable_stops_submissions_but_capture_and_failure_artifacts_continue(monkeypatch, tmp_path):
    frames = [packet(index, index * 30) for index in range(1, 5)]
    unavailable = DetectionResult(1, "gemini", "unavailable", model="offline-test-model", error="Quota unavailable")
    worker = FakeWorker([None, (frames[0], unavailable), None, None])
    install_worker(monkeypatch, worker)
    capture, manager, open_source = install_capture(frames)
    shown = install_preview(monkeypatch, len(frames))
    settings = settings_for(tmp_path)

    result = live.run_live_detection(settings, object(), open_source=open_source, gemini_live=True)

    assert worker.submissions == [1] and worker.closed
    assert capture.read_packet.call_count == len(shown) == 4
    assert result.frames_processed == 4 and result.ready_frames == 0 and result.not_ready_frames == 1
    assert result.last_inference.status == "unavailable" and result.detections == ()
    manager.__exit__.assert_called_once()
    report = json.loads(result.report_path.read_text())
    assert report["cloud_uploads_paused"] is True
    assert report["cloud_timeout_s"] == 60
    assert report["inference_results_received"] == 1
    assert report["output_frame_id"] == report["last_result_frame_id"] == 1
    assert report["last_analyzed_frame_id"] is None
    assert report["last_preview_frame_id"] == 4
    assert report["last_inference_status"] == "unavailable"
    assert report["last_inference"]["error"] == "Quota unavailable"
    assert int(cv2.imread(str(result.output_path))[60, 100, 0]) == 30


def test_analysis_history_keeps_latest_120_of_125_results_and_all_camera_counts(monkeypatch, tmp_path):
    frames = [packet(index) for index in range(1, 127)]
    worker = FakeWorker([None] + [(frame, ready(frame)) for frame in frames[:-1]])
    install_worker(monkeypatch, worker)
    capture, manager, open_source = install_capture([*frames, KeyboardInterrupt])
    settings = settings_for(tmp_path, preview=False, output=False)

    result = live.run_live_detection(settings, object(), open_source=open_source, gemini_live=True)

    assert result.interrupted and worker.closed
    assert result.frames_processed == 126 and result.ready_frames == 125
    assert capture.read_packet.call_count == 127 and len(worker.submissions) == 126
    manager.__exit__.assert_called_once()
    report = json.loads(result.report_path.read_text())
    assert report["frames_processed"] == 126 and report["inference_results_received"] == 125
    assert report["frame_records_limit"] == len(report["frames"]) == 120
    assert report["frame_records_truncated"] and report["frame_records_dropped"] == 5
    assert report["frames"][0]["frame_id"] == 6 and report["frames"][-1]["frame_id"] == 125
    assert all(record["frame_id"] == record["inference"]["frame_id"] for record in report["frames"])
    assert report["frame_records_scope"] == "received_inference_results_only"
    assert report["last_preview_frame_id"] == 126 and report["last_analyzed_frame_id"] == 125


def test_live_profile_needs_explicit_cloud_opt_in_before_worker_or_camera(monkeypatch):
    settings = load_settings(ROOT / "config/pi-live.yaml")
    assert settings.camera.source == "picamera2" and settings.camera.fps == 30.0
    assert settings.simulation.async_detection and settings.simulation.preview
    assert settings.models.object_detection_backend == "gemini"
    assert settings.cloud.live_interval_s >= 5.0
    open_source = Mock(side_effect=AssertionError("Camera opened without cloud opt-in"))
    worker_factory = Mock(side_effect=AssertionError("Worker constructed without cloud opt-in"))
    monkeypatch.setattr(live, "LiveDetector", worker_factory)

    with pytest.raises(ConfigurationError, match="Gemini detection requires"):
        live.run_live_detection(settings, object(), open_source=open_source)

    worker_factory.assert_not_called()
    open_source.assert_not_called()


@pytest.mark.parametrize("status, expected, forbidden", [
    ("pending", "Gemini request pending", "No objects identified"),
    ("unavailable", "Gemini request timed out", "No objects identified"),
    ("ready", "No objects identified", "No detection result received"),
])
def test_request_states_distinguish_timeout_and_pending_from_successful_empty_result(monkeypatch, status, expected, forbidden):
    current, selected = packet(8), packet(7)
    text = Mock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", text)
    attempted = None if status == "pending" else (
        selected, DetectionResult(7, "gemini", status, error="Gemini timed out." if status == "unavailable" else None),
    )

    live.render_live_preview(current, attempted, paused=status == "unavailable", request_timeout_s=60)

    captions = [call.args[1] for call in text.call_args_list]
    assert expected in captions and forbidden not in captions
    if status == "pending":
        assert "Request timeout: 60s" in captions
    elif status == "unavailable":
        assert "No detection result received" in captions
        assert any(caption.startswith("ATTEMPTED frame 7") for caption in captions)
        assert not any(caption.startswith("ANALYZED") for caption in captions)
