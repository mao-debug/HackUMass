"""Chess UI integration without camera, network, microphone, or audio devices."""

from dataclasses import dataclass
import json
from pathlib import Path
import socket
import threading
import time as real_time
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.core import chess_live
from vidi.core.chess_types import BoardObservation
from vidi.core.configuration import CameraSettings, CloudSettings, ModelSettings, SimulationSettings, SpeechSettings
from vidi.core.types import FramePacket, SpeechResult

REAL_TASK = chess_live._Task
REAL_SESSION_FACTORY = chess_live._create_session
REAL_OBSERVER_FACTORY = chess_live._create_observer


class Clock:
    now = 10.

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(self.now * 1e9)

    def sleep(self, _seconds):
        self.now += .05


class Camera:
    def __init__(self, clock, advances=None):
        self.clock, self.advances = clock, advances or {}
        self.count, self.closed = 0, False
        self.buffer = np.zeros((120, 160, 3), np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def read(self):
        self.count += 1
        self.clock.now += self.advances.get(self.count, 0.)
        self.buffer[:] = self.count
        return self.buffer


class ImmediateTask:
    def __init__(self, operation):
        self.cancel, self.done = threading.Event(), threading.Event()
        self.failed, self.failure = False, None
        try:
            self.value = operation(self.cancel)
        except BaseException as error:
            self.failed, self.failure, self.value = True, error, None
        self.done.set()

    def close(self):
        self.cancel.set()


@dataclass(frozen=True)
class Transition:
    status: str
    message: str
    proposal_id: str | None = None
    uci: str | None = None


class Session:
    """A rules-boundary fake; rules legality is covered by session's own tests."""

    def __init__(self):
        self.pending = None
        self.initialized = False
        self.moves = []
        self.proposed, self.confirmed, self.queries = [], [], []

    def propose(self, observation, now_ns):
        self.proposed.append(observation)
        if observation.status != "ready":
            return Transition("invalid_observation", "Board is unclear; confirmed history is unchanged.")
        self.pending = SimpleNamespace(proposal_id=str(observation.frame_id), captured_monotonic_ns=observation.captured_monotonic_ns,
                                       uci="e2e4" if self.initialized else None)
        return Transition("proposed", "Possible starting position." if not self.initialized else "Possible white pawn from e2 to e4.",
                          self.pending.proposal_id, self.pending.uci)

    def confirm(self, proposal_id, now_ns):
        self.confirmed.append(proposal_id)
        if now_ns - self.pending.captured_monotonic_ns > 30_000_000_000:
            self.pending = None
            return Transition("stale", "Proposal expired. Scan again.")
        self.initialized = True
        if self.pending.uci:
            self.moves.append(self.pending.uci)
        self.pending = None
        return Transition("confirmed", "Position confirmed.")

    def reject(self):
        existed = self.pending is not None
        self.pending = None
        return Transition("rejected" if existed else "no_pending", "Proposal discarded; confirmed history is unchanged.")

    def query(self, action, square=None):
        self.queries.append((action, square))
        return "Confirmed history is unavailable until the starting position is confirmed." if not self.initialized else "White pawn on e4."

    def snapshot(self):
        return dict(initialized=self.initialized, fen="fixture-position" if self.initialized else None,
                    confirmed_moves=self.moves[:], pending=None if self.pending is None else {"proposal_id": self.pending.proposal_id})


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    touched = []

    def forbidden(*_args, **_kwargs):
        touched.append(True)
        raise AssertionError("Unmocked chess hardware/service boundary")

    for name in ("_create_observer", "_record_command", "_transcribe_command", "LocalSpeech"):
        monkeypatch.setattr(chess_live, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(chess_live, "_Task", ImmediateTask)
    session, clock = Session(), Clock()
    monkeypatch.setattr(chess_live, "_create_session", lambda _: session)
    monkeypatch.setattr(chess_live, "time", clock)
    yield SimpleNamespace(session=session, clock=clock)
    assert not touched


def settings(tmp_path, backend="gemini", output=False):
    return SimpleNamespace(camera=CameraSettings(source="sample", width=160, height=120, fps=30),
                           simulation=SimulationSettings(preview=True, report_path=tmp_path / "chess.json",
                                                         output_path=tmp_path / "board.png" if output else None),
                           models=ModelSettings(), cloud=CloudSettings(), speech=SpeechSettings(),
                           voice=SimpleNamespace(device="pulse", record_seconds=5, timeout_s=30, model=None, audio_path=None),
                           chess=SimpleNamespace(backend=backend, orientation="white_near", observation_max_age_s=30,
                                                 fixture_path=Path(__file__).resolve().parents[1] / "examples" / "chess-fixture.json"))


def keys(monkeypatch, sequence):
    iterator = iter(sequence)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(iterator))


def observer(monkeypatch, callback=None):
    observe = Mock(side_effect=callback or (lambda packet, **kwargs: BoardObservation(
        packet.frame_id, packet.captured_monotonic_ns, status="ready", squares=(".",) * 64, model="fake-model")))
    factory = Mock(return_value=SimpleNamespace(observe=observe))
    monkeypatch.setattr(chess_live, "_create_observer", factory)
    return factory, observe


def speaker(monkeypatch, callback=None):
    speak = Mock(side_effect=callback or (lambda text, **kwargs: SpeechResult(status="completed", text=text)))
    monkeypatch.setattr(chess_live, "LocalSpeech", Mock(return_value=SimpleNamespace(speak=speak)))
    return speak


def voice(monkeypatch, text="check board", status="ready"):
    record = Mock(return_value=object())
    transcribe = Mock(return_value=SimpleNamespace(status=status, text=text, model="fake-audio", elapsed_s=.1, error=None))
    monkeypatch.setattr(chess_live, "_record_command", record)
    monkeypatch.setattr(chess_live, "_transcribe_command", transcribe)
    return record, transcribe


def run(config, camera, *, speak=False, voice_commands=False):
    return chess_live.run_chess_live(config, speak=speak, voice_commands=voice_commands, open_source=lambda _: camera)


@pytest.mark.parametrize("pending", [False, True])
def test_spoken_turn_query_labels_confirmed_state_and_pending_observation(monkeypatch, tmp_path, isolated, pending):
    isolated.session.initialized = True
    if pending:
        isolated.session.pending = SimpleNamespace(proposal_id="unconfirmed-view")
    speech = speaker(monkeypatch)
    keys(monkeypatch, [ord("w"), -1, ord("q")])
    run(settings(tmp_path), Camera(isolated.clock), speak=True)
    text = speech.call_args.args[0]
    assert text.startswith("From the last confirmed position:")
    assert ("still awaits confirmation" in text) is pending


def test_no_scan_does_not_create_observer_or_capture_audio(monkeypatch, tmp_path, isolated):
    keys(monkeypatch, [-1, ord("q")])
    camera = Camera(isolated.clock)
    result = run(settings(tmp_path), camera, voice_commands=True)
    report = json.loads(result.report_path.read_text())
    assert camera.closed and not report["session"]["initialized"]
    assert report["schema_version"] == 5 and report["action_totals"]["cloud_requests"] == 0


def test_scan_owns_exact_raw_frame_and_never_commits_without_c(monkeypatch, tmp_path, isolated):
    config, camera = settings(tmp_path, output=True), Camera(isolated.clock)
    config.camera = CameraSettings(source="sample", width=1280, height=960, fps=30)
    camera.buffer = np.zeros((960, 1280, 3), np.uint8)

    def observe(packet, *, cancel_event):
        report = json.loads(config.simulation.report_path.read_text())
        assert report["status"] == "pending" and not report["session"]["initialized"]
        assert packet.frame_id == 1 and packet.image_bgr[100, 100, 0] == 1
        assert packet.image_bgr.shape == (960, 1280, 3)
        assert not np.shares_memory(packet.image_bgr, camera.buffer) and not cancel_event.is_set()
        return BoardObservation(packet.frame_id, packet.captured_monotonic_ns, "ready", (".",) * 64)

    factory, observed = observer(monkeypatch, observe)
    keys(monkeypatch, [ord("b"), -1, ord("w"), ord("q")])
    result = run(config, camera)
    factory.assert_called_once()
    observed.assert_called_once()
    assert len(isolated.session.proposed) == 1 and not isolated.session.confirmed
    assert not isolated.session.initialized
    assert int(cv2.imread(str(result.output_path))[100, 100, 0]) == 1
    assert json.loads(result.report_path.read_text())["observation"]["frame_id"] == 1


def test_b_with_pending_fixture_does_not_advance_observer_or_replace_proposal(monkeypatch, tmp_path, isolated):
    pytest.importorskip("chess")
    monkeypatch.setattr(chess_live, "_create_session", REAL_SESSION_FACTORY)
    factory = Mock(wraps=REAL_OBSERVER_FACTORY)
    monkeypatch.setattr(chess_live, "_create_observer", factory)
    # The second B must not consume the next invented position before C.
    keys(monkeypatch, [ord("b"), ord("b"), ord("c"), ord("b"), ord("c"), ord("q")])
    result = run(settings(tmp_path, backend="fixture"), Camera(isolated.clock))
    report = json.loads(result.report_path.read_text())
    assert report["session"]["initialized"] and report["session"]["confirmed_moves"] == ["e2e4"]
    assert report["action_totals"]["scan_requests"] == 2
    assert any(item["action"] == "scan" and item["status"] == "ignored_pending" for item in report["actions"])


def test_voice_scan_with_existing_proposal_cannot_request_or_replace_an_image(monkeypatch, tmp_path, isolated):
    _, observed = observer(monkeypatch)
    voice(monkeypatch, "check board")
    camera = Camera(isolated.clock, {3: 6.})

    def key(_delay):
        if camera.count == 1:
            return ord("b")
        if camera.count == 3:
            return ord("m")
        if camera.count == 6:
            assert isolated.session.pending.proposal_id == "1"
            return ord("q")
        return -1

    monkeypatch.setattr(cv2, "waitKey", key)
    result = run(settings(tmp_path), camera, voice_commands=True)
    observed.assert_called_once()
    report = json.loads(result.report_path.read_text())
    assert report["action_totals"]["scan_requests"] == 1
    assert report["action_totals"]["audio_requests"] == 1
    assert report["observation"]["frame_id"] == 1
    assert any(item["status"] == "ignored_pending" for item in report["actions"])


def test_real_fixture_and_rules_require_separate_confirmations_for_start_and_move(monkeypatch, tmp_path, isolated):
    pytest.importorskip("chess")
    monkeypatch.setattr(chess_live, "_create_session", REAL_SESSION_FACTORY)
    monkeypatch.setattr(chess_live, "_create_observer", REAL_OBSERVER_FACTORY)
    spoken = speaker(monkeypatch)
    keys(monkeypatch, [ord("b"), ord("c"), ord("b"), ord("c"), ord("w"), ord("q")])
    result = run(settings(tmp_path, backend="fixture"), Camera(isolated.clock), speak=False)
    report = json.loads(result.report_path.read_text())
    assert report["session"]["initialized"] and report["session"]["confirmed_moves"] == ["e2e4"]
    assert report["action_totals"]["scan_requests"] == 2 and report["action_totals"]["cloud_requests"] == 0
    assert result.simulated and result.description.startswith("SIMULATED.")
    spoken.assert_not_called()


def test_real_fixture_proposal_speech_is_compact_explicit_and_labeled_once(monkeypatch, tmp_path, isolated):
    pytest.importorskip("chess")
    monkeypatch.setattr(chess_live, "_create_session", REAL_SESSION_FACTORY)
    monkeypatch.setattr(chess_live, "_create_observer", REAL_OBSERVER_FACTORY)
    spoken = speaker(monkeypatch)
    keys(monkeypatch, [ord("b"), -1, -1, ord("q")])
    result = run(settings(tmp_path, backend="fixture"), Camera(isolated.clock), speak=True)
    spoken.assert_called_once()
    text = spoken.call_args.args[0]
    assert text == "SIMULATED. Proposed: Standard starting position. Partner: verify, then C confirms or X rejects."
    assert text.count("SIMULATED") == 1 and "Simulated observation" not in text
    assert len(text.split()) < 25
    assert not json.loads(result.report_path.read_text())["session"]["initialized"]


def test_fixture_voice_is_rejected_before_any_camera_or_session(monkeypatch, tmp_path):
    from vidi.core.controller import SimulationError
    monkeypatch.setattr(chess_live, "_create_session", lambda _: pytest.fail("Session created"))
    with pytest.raises(SimulationError, match="cannot enable"):
        chess_live.run_chess_live(settings(tmp_path, backend="fixture"), speak=False, voice_commands=True,
                                  open_source=lambda _: pytest.fail("Camera opened"))


def test_c_is_explicit_and_s_discards_a_pending_proposal(monkeypatch, tmp_path, isolated):
    observer(monkeypatch)
    keys(monkeypatch, [ord("b"), ord("s"), ord("c"), ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock))
    assert not isolated.session.confirmed and not isolated.session.initialized
    assert json.loads(result.report_path.read_text())["session"]["pending"] is None


def test_stale_confirm_does_not_advance_position(monkeypatch, tmp_path, isolated):
    observer(monkeypatch)
    keys(monkeypatch, [ord("b"), -1, ord("c"), ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock, {3: 31.}))
    assert not isolated.session.initialized
    assert json.loads(result.report_path.read_text())["transition"]["status"] == "stale"


def test_unclear_observation_abstains_and_queries_use_confirmed_session_only(monkeypatch, tmp_path, isolated):
    observer(monkeypatch, lambda packet, **kwargs: BoardObservation(packet.frame_id, packet.captured_monotonic_ns, status="unclear"))
    keys(monkeypatch, [ord("b"), ord("c"), ord("w"), ord("l"), ord("a"), ord("h"), ord("q")])
    run(settings(tmp_path), Camera(isolated.clock))
    assert not isolated.session.confirmed
    assert isolated.session.queries == [("turn", None), ("last_move", None), ("board", None)]


def test_failed_speech_does_not_destroy_observation_or_confirmed_transition(monkeypatch, tmp_path, isolated):
    observer(monkeypatch)
    spoken = speaker(monkeypatch, lambda text, **kwargs: SpeechResult(status="failed", error="Engine unavailable."))
    keys(monkeypatch, [ord("b"), -1, -1, ord("c"), -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock), speak=True)
    assert isolated.session.initialized and spoken.call_count == 2
    report = json.loads(result.report_path.read_text())
    assert report["transition"]["status"] == "confirmed" and report["observation"]["status"] == "ready"
    assert report["speech"]["status"] == "failed"


def test_c_while_proposal_speech_is_busy_is_explicitly_ignored_not_queued(monkeypatch, tmp_path, isolated):
    observer(monkeypatch)
    speaker(monkeypatch)
    keys(monkeypatch, [ord("b"), ord("c"), ord("c"), -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock), speak=True)
    report = json.loads(result.report_path.read_text())
    assert len(isolated.session.confirmed) == 1
    assert any(item["action"] == "confirm" and item["status"] == "ignored_busy" for item in report["actions"])


@pytest.mark.parametrize("cancel_key", [ord("s"), ord("q")])
def test_cancel_after_recording_prevents_audio_upload(monkeypatch, tmp_path, isolated, cancel_key):
    recorded, transcribed = voice(monkeypatch)
    keys(monkeypatch, [ord("m"), cancel_key] + ([] if cancel_key == ord("q") else [-1, ord("q")]))
    result = run(settings(tmp_path), Camera(isolated.clock), voice_commands=True)
    recorded.assert_called_once()
    transcribed.assert_not_called()
    report = json.loads(result.report_path.read_text())
    assert report["action_totals"]["cloud_requests"] == 0
    assert report["voice_status"] == "cancelled" and report["status"] != "pending"


@pytest.mark.parametrize("cancel_key", [ord("s"), ord("q"), ord("x")])
def test_cancel_after_transcription_discards_waiting_scan(monkeypatch, tmp_path, isolated, cancel_key):
    _, transcribed = voice(monkeypatch)
    keys(monkeypatch, [ord("m"), -1, cancel_key] + ([] if cancel_key == ord("q") else [-1, ord("q")]))
    result = run(settings(tmp_path), Camera(isolated.clock, {4: 6.}), voice_commands=True)
    transcribed.assert_called_once()
    assert not isolated.session.proposed
    assert json.loads(result.report_path.read_text())["action_totals"]["cloud_requests"] == 1


def test_voice_scan_waits_shared_cooldown_and_owns_newer_frame(monkeypatch, tmp_path, isolated, caplog):
    raw = "VIDI, PLEASE CHECK THE BOARD!"
    recorded, transcribed = voice(monkeypatch, raw)
    _, observed = observer(monkeypatch)
    keys(monkeypatch, [ord("m"), -1, -1, -1, -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock, {4: 5.1}), voice_commands=True)
    recorded.assert_called_once()
    transcribed.assert_called_once()
    observed.assert_called_once()
    assert observed.call_args.args[0].frame_id == 4
    assert len(isolated.session.proposed) == 1 and not isolated.session.confirmed
    report = json.loads(result.report_path.read_text())
    assert report["action_totals"]["cloud_requests"] == 2
    assert raw not in result.report_path.read_text() + caplog.text


@pytest.mark.parametrize("status,text", [("ready", "move my queen to h5"), ("no_speech", ""), ("failed", "PRIVATE")])
def test_unknown_silent_failed_voice_never_moves_or_guesses(monkeypatch, tmp_path, isolated, status, text):
    voice(monkeypatch, text, status)
    keys(monkeypatch, [ord("m"), -1, -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock), voice_commands=True)
    assert not isolated.session.proposed and not isolated.session.confirmed
    report = json.loads(result.report_path.read_text())
    assert report["requests_paused"] is (status == "failed")
    assert not text or text not in result.report_path.read_text()


def test_voice_piece_query_is_local_after_audio_and_uses_only_confirmed_state(monkeypatch, tmp_path, isolated):
    voice(monkeypatch, "what is on e four")
    keys(monkeypatch, [ord("m"), -1, -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock), voice_commands=True)
    assert isolated.session.queries == [("piece", "e4")]
    assert not isolated.session.initialized
    assert json.loads(result.report_path.read_text())["action_totals"]["scan_requests"] == 0


def test_spoken_confirm_is_a_separate_explicit_action_after_scan(monkeypatch, tmp_path, isolated):
    observer(monkeypatch)
    voice(monkeypatch, "confirm position")
    keys(monkeypatch, [ord("b"), -1, ord("m"), -1, -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock, {3: 6.}), voice_commands=True)
    assert isolated.session.initialized and isolated.session.confirmed == ["1"]
    report = json.loads(result.report_path.read_text())
    assert report["action_totals"]["scan_requests"] == 1
    assert report["action_totals"]["audio_requests"] == 1


@pytest.mark.parametrize("mismatch", ["frame", "timestamp", "simulation"])
def test_wrong_request_identity_never_reaches_game_proposal(monkeypatch, tmp_path, isolated, mismatch):
    def wrong(packet, **kwargs):
        return BoardObservation(packet.frame_id + (mismatch == "frame"),
                                packet.captured_monotonic_ns + (mismatch == "timestamp"),
                                "ready", (".",) * 64, simulated=mismatch == "simulation")

    observer(monkeypatch, wrong)
    keys(monkeypatch, [ord("b"), -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock))
    assert not isolated.session.proposed
    report = json.loads(result.report_path.read_text())
    assert report["observation"]["status"] == "failed" and report["requests_paused"]


def test_failed_scan_pauses_until_r_but_local_queries_work(monkeypatch, tmp_path, isolated):
    _, observed = observer(monkeypatch, lambda packet, **kwargs: BoardObservation(packet.frame_id, packet.captured_monotonic_ns, status="failed"))
    keys(monkeypatch, [ord("b"), ord("b"), ord("w"), ord("r"), ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock))
    observed.assert_called_once()
    assert isolated.session.queries == [("turn", None)]
    assert not json.loads(result.report_path.read_text())["requests_paused"]


def test_cancelled_deferred_scan_cannot_construct_observer_or_propose(monkeypatch, tmp_path, isolated):
    jobs = []

    class Deferred:
        def __init__(self, operation):
            self.operation, self.cancel, self.done = operation, threading.Event(), threading.Event()
            self.value, self.failed = None, False
            jobs.append(self)

        def close(self):
            self.cancel.set()

    monkeypatch.setattr(chess_live, "_Task", Deferred)
    camera = Camera(isolated.clock)

    def key(_delay):
        if camera.count == 1:
            return ord("b")
        if camera.count == 2:
            return ord("s")
        if camera.count == 3:
            assert jobs[0].cancel.is_set()
            jobs[0].value = jobs[0].operation(jobs[0].cancel)
            jobs[0].done.set()
            return -1
        return ord("q")

    monkeypatch.setattr(cv2, "waitKey", key)
    run(settings(tmp_path), camera)
    assert not isolated.session.proposed and camera.closed


def test_quit_records_cancellation_and_preview_uses_exact_pending_scan(monkeypatch, tmp_path, isolated):
    class Deferred:
        def __init__(self, operation):
            self.cancel, self.done = threading.Event(), threading.Event()
            self.value, self.failed = None, False

        def close(self):
            self.cancel.set()

    monkeypatch.setattr(chess_live, "_Task", Deferred)
    rendered = Mock(wraps=chess_live.render_chess_preview)
    monkeypatch.setattr(chess_live, "render_chess_preview", rendered)
    keys(monkeypatch, [ord("b"), -1, ord("q")])
    result = run(settings(tmp_path), Camera(isolated.clock))
    assert rendered.call_args_list[1].args[1].frame_id == 1
    assert rendered.call_args_list[1].kwargs["scan_pending"] is True
    report = json.loads(result.report_path.read_text())
    assert report["status"] == "cancelled" and report["session"]["pending"] is None
    assert any(item["action"] == "scan" and item["status"] == "cancelled" for item in report["actions"])


def test_live_preview_and_cleanup_continue_while_late_scan_result_is_cancelled(monkeypatch, tmp_path, isolated):
    entered, released = threading.Event(), threading.Event()
    monkeypatch.setattr(chess_live, "_Task", REAL_TASK)
    monkeypatch.setattr(chess_live, "time", SimpleNamespace(monotonic=real_time.monotonic,
                        monotonic_ns=real_time.monotonic_ns, sleep=lambda _: real_time.sleep(.001)))
    camera = Camera(isolated.clock)

    def wait_for_cancel(packet, *, cancel_event):
        entered.set()
        assert cancel_event.wait(1.)
        assert not camera.closed
        released.set()
        return BoardObservation(packet.frame_id, packet.captured_monotonic_ns, "ready", (".",) * 64)

    _, observed = observer(monkeypatch, wait_for_cancel)

    def key(_delay):
        if camera.count == 1:
            return ord("b")
        assert entered.wait(.2)
        if camera.count == 5:
            return ord("s")
        if camera.count > 5 and released.is_set():
            return ord("q")
        assert camera.count < 100
        return -1

    monkeypatch.setattr(cv2, "waitKey", key)
    result = run(settings(tmp_path), camera)
    observed.assert_called_once()
    assert released.is_set() and camera.closed and result.frames_processed >= 6
    assert not isolated.session.proposed and not isolated.session.confirmed


def test_simulated_feedback_is_labeled_and_action_history_is_bounded(monkeypatch, tmp_path, isolated):
    spoken = speaker(monkeypatch)
    count = 0

    def key(_delay):
        nonlocal count
        count += 1
        return ord("h") if count < 130 else ord("q")

    monkeypatch.setattr(cv2, "waitKey", key)
    result = run(settings(tmp_path, backend="fixture"), Camera(isolated.clock), speak=True)
    assert all(call.args[0].startswith("SIMULATED.") for call in spoken.call_args_list)
    report = json.loads(result.report_path.read_text())
    assert len(report["actions"]) == 120 and report["action_records_truncated"]
    assert report["action_totals"]["cloud_requests"] == 0


def test_preview_does_not_modify_live_or_scanned_input():
    packet = FramePacket(8, 10_000_000_000, np.full((240, 320, 3), 40, np.uint8))
    observed = FramePacket(1, 9_000_000_000, np.full((240, 320, 3), 120, np.uint8))
    display = chess_live.render_chess_preview(packet, observed, {"initialized": False, "pending": None}, "SIMULATED.",
                                             status="not_requested", voice_status="not_requested", paused=False,
                                             simulated=True, voice_commands=False)
    assert np.all(packet.image_bgr == 40) and np.all(observed.image_bgr == 120)
    np.testing.assert_array_equal(display[220, 100], (40, 40, 40))
    np.testing.assert_array_equal(display[220, 420], (120, 120, 120))


def test_large_preview_is_bounded_preserves_aspect_and_hides_technical_fen(monkeypatch):
    packet = FramePacket(8, 10_000_000_000, np.full((720, 1280, 3), 40, np.uint8))
    observed = FramePacket(1, 9_000_000_000, np.full((960, 1280, 3), 120, np.uint8))
    drawn, original = [], cv2.putText

    def draw(image, text, *args):
        drawn.append(text)
        return original(image, text, *args)

    monkeypatch.setattr(cv2, "putText", draw)
    display = chess_live.render_chess_preview(packet, observed,
        {"initialized": True, "pending": None, "confirmed_moves": ["e2e4"],
         "last_move": "White pawn from e2 to e4", "fen": "TECHNICAL-FEN-ONLY-IN-REPORT"},
        "Confirmed move.", status="confirmed", voice_status="not_requested", paused=False,
        simulated=False, voice_commands=False)
    assert display.shape == (480, 1280, 3)
    assert np.all(packet.image_bgr == 40) and np.all(observed.image_bgr == 120)
    np.testing.assert_array_equal(display[300, 100], (40, 40, 40))
    np.testing.assert_array_equal(display[450, 100], (0, 0, 0))  # 16:9 remains letterboxed.
    np.testing.assert_array_equal(display[450, 740], (120, 120, 120))
    assert "Confirmed moves: 1" in " ".join(drawn)
    assert "TECHNICAL-FEN" not in " ".join(drawn)
