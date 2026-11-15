"""Headless event/cancellation regressions; no real devices or service clients."""

from dataclasses import replace
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vidi.assistant.intent_router import Intent, route_intent
from vidi.core.assistant_runtime import AssistantController
from vidi.core.configuration import AppSettings, ChessSettings
from vidi.core.types import FramePacket, SpeechResult


class Broker:
    frames = 1
    first_ns = last_ns = 1
    error = None

    def __init__(self):
        self.paused = self.closed = False

    def start(self):
        return self

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def close(self):
        self.closed = True


class Speaker:
    def __init__(self):
        self.words = []
        self.stops = 0

    def speak(self, text, **kwargs):
        if not kwargs["cancel_event"].is_set():
            self.words.append(text)
        return SpeechResult(status="completed", backend="test")

    def stop(self):
        self.stops += 1

    def close(self):
        pass


class Listener:
    def __init__(self):
        self.paused = self.closed = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def close(self):
        self.closed = True


def make_controller(**options):
    return AssistantController(AppSettings(), speaker=Speaker(), broker=Broker(), **options)


def settle(controller):
    deadline = time.monotonic() + 2
    while (controller.job is not None or controller.speech_job is not None) and time.monotonic() < deadline:
        controller.tick()
        time.sleep(.001)
    assert controller.job is None


@pytest.mark.parametrize(("text", "mode", "action"), [
    ("Hey vidi, what moved?", "chess", "scan"),
    ("hey vidi chess mode", "normal", "mode_chess"),
    ("switch to normal mode", "chess", "mode_normal"),
    ("read the text in front of me", "normal", "read_text"),
    ("capture an image", "normal", "capture"),
    ("don't capture an image", "normal", "unknown"),
    ("capture an image and describe it", "normal", "unknown"),
    ("what is directly in front of me", "normal", "describe"),
])
def test_whole_utterance_intents(text, mode, action):
    assert route_intent(text, mode).action == action


def test_late_completion_after_wake_cannot_speak_or_commit():
    controller = make_controller()
    started, finish = threading.Event(), threading.Event()

    def work(cancel):
        started.set()
        finish.wait(1)
        return "obsolete reply"

    controller._launch("capture", work)
    assert started.wait(1)
    controller.post("wake")
    controller.tick()
    finish.set()
    settle(controller)
    assert "obsolete reply" not in controller.speaker.words
    controller.close()


def test_privacy_pause_closes_inputs_and_requires_physical_resume():
    listener = Listener()
    controller = make_controller(listener=listener)
    controller.handle("privacy_pause")
    assert controller.privacy_paused and listener.paused and controller.broker.paused
    controller.handle("capture")
    assert controller.job is None
    controller.handle("privacy_toggle")
    assert not controller.privacy_paused and not listener.paused and not controller.broker.paused
    controller.close()


def test_queue_saturation_does_not_lose_stop_or_keep_pending():
    controller = make_controller(mode="chess")
    controller._ensure_chess()
    from vidi.core.chess_types import BoardObservation
    import chess
    now = time.monotonic_ns()
    board = chess.Board()
    squares = tuple(p.symbol() if (p := board.piece_at(i)) else "." for i in range(64))
    assert controller.session.propose(BoardObservation(1, now, "ready", squares)).status == "proposed"
    for _ in range(16):
        controller.post("unknown")
    controller.post("stop")
    controller.tick()
    assert controller.session.pending is None
    controller.close()


def test_wake_preserves_existing_manual_proposal_for_explicit_voice_confirmation(monkeypatch):
    controller = make_controller(mode="chess", cloud_enabled=True)
    controller._ensure_chess()
    from vidi.core.chess_types import BoardObservation
    import chess
    board = chess.Board()
    squares = tuple(p.symbol() if (p := board.piece_at(i)) else "." for i in range(64))
    controller.session.propose(BoardObservation(1, time.monotonic_ns(), "ready", squares))
    proposal = controller.session.pending.proposal_id
    monkeypatch.setattr(controller, "_record_command", lambda cancel: SimpleNamespace(status="ready", text="confirm position"))
    controller.post("wake")
    settle(controller) if controller.events.empty() else controller.tick()
    settle(controller)
    assert controller.session.initialized
    assert controller.session.pending is None
    assert proposal
    controller.close()


def test_mode_switch_preserves_confirmed_history_but_requires_fresh_scan():
    controller = make_controller()
    controller.handle("mode_chess")
    session = controller.session
    controller.needs_chess_scan = False
    controller.handle("mode_normal")
    controller.handle("mode_chess")
    assert controller.session is session
    assert controller.needs_chess_scan
    controller.handle("turn")
    assert "fresh board scan" in controller.last_reply
    controller.close()


def test_no_cloud_client_is_constructed_without_opt_in(monkeypatch):
    from vidi.ai import scene_description
    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", lambda *_: pytest.fail("unexpected client"))
    controller = make_controller()
    controller.handle("describe")
    assert controller.job is None
    controller.post("wake")
    controller.tick()
    assert controller.totals["cloud_requests"] == 0
    controller.close()


def test_paired_provider_failure_pauses_uploads_and_preserves_state():
    controller = make_controller(mode="chess", cloud_enabled=True)
    controller._ensure_chess()
    from vidi.core.chess_types import BoardObservation
    observations = (BoardObservation(1, time.monotonic_ns(), "failed"),
                    BoardObservation(2, time.monotonic_ns(), "failed"))
    controller._scan_generation = controller.automatic.generation
    controller._launch("scan", lambda cancel: observations)
    settle(controller)
    assert controller.cloud_paused
    assert not controller.session.initialized
    controller.close()


def test_one_cloud_request_at_a_time_even_after_cancel():
    controller = make_controller(cloud_enabled=True)
    held, release, second_started = threading.Event(), threading.Event(), threading.Event()
    first_cancel, second_cancel = threading.Event(), threading.Event()

    def first():
        held.set()
        release.wait(1)

    a = threading.Thread(target=lambda: controller._cloud(first, first_cancel))
    b = threading.Thread(target=lambda: controller._cloud(second_started.set, second_cancel))
    a.start()
    assert held.wait(1)
    first_cancel.set()
    b.start()
    assert not second_started.wait(.05)
    release.set()
    a.join(1)
    b.join(1)
    assert second_started.is_set()
    controller.close()


def test_cancelled_photo_does_not_write(monkeypatch, tmp_path):
    controller = make_controller()
    controller.settings = replace(controller.settings, assistant=replace(controller.settings.assistant, photo_dir_path=tmp_path))
    release = threading.Event()

    def fresh(cancel, **kwargs):
        release.wait(1)
        return FramePacket(1, time.monotonic_ns(), np.zeros((20, 20, 3), np.uint8))

    monkeypatch.setattr(controller, "_fresh", fresh)
    controller.handle("capture")
    controller.post("stop")
    controller.tick()
    release.set()
    for task in controller.retired:
        task.thread.join(1)
    assert not list(tmp_path.iterdir())
    controller.close()


def test_reports_do_not_include_transcripts_audio_or_utterances():
    controller = make_controller()
    controller._say("private scene words")
    settle(controller)
    import json
    report = json.dumps(controller.snapshot())
    assert "private scene words" not in report
    assert "wav_bytes" not in report
    assert "transcript" not in report
    controller.close()


def test_actual_headless_loop_reads_sample_frames_without_a_window(monkeypatch, tmp_path):
    from vidi.core import keyboard_control
    from vidi.core.assistant_runtime import run_assistant
    import cv2

    class Keyboard:
        error = None

        def __init__(self, path, callback):
            pass

        def start(self):
            return self

        def close(self):
            pass

    monkeypatch.setattr(keyboard_control, "KeyboardControl", Keyboard)
    monkeypatch.setattr(cv2, "imshow", lambda *args: pytest.fail("headless created window"))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda *args: pytest.fail("headless polled window"))
    original = AppSettings()
    settings = replace(original, assistant=replace(original.assistant, keyboard_device_path=tmp_path / "keyboard"),
                       simulation=replace(original.simulation, report_path=tmp_path / "report.json"))
    result = run_assistant(settings, wake=False, speak=False, run_seconds=.1)
    assert result["frames_processed"] >= 1
    assert result["totals"]["cloud_requests"] == 0
    assert (tmp_path / "report.json").is_file()


def test_failed_headset_resume_stays_paused_without_ending_the_app():
    class Missing(Listener):
        def resume(self):
            raise OSError("private device error")

    listener = Missing()
    controller = make_controller(listener=listener)
    controller.handle("privacy_pause")
    controller.handle("privacy_toggle")
    assert controller.privacy_paused and listener.paused and controller.broker.paused
    assert "Input is still unavailable" in controller.last_reply
    assert "private" not in controller.last_reply
    controller.close()
