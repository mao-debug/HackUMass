"""Offline preview cannot use credentials, cloud actions, or a microphone."""

from dataclasses import replace
import json
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.core import controller, phase4
from vidi.core.configuration import (
    AppSettings, CameraSettings, ConfigurationError, ModelSettings, ProximitySettings,
    SimulationSettings, VoiceSettings, validate_runtime_settings,
)
from vidi.core.types import FramePacket, SpeechResult
import vidi.main as cli


class Camera:
    def __init__(self):
        self.count, self.closed = 0, False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def read(self):
        self.count += 1
        return np.full((120, 160, 3), self.count, np.uint8)


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


@pytest.fixture(autouse=True)
def no_external_boundaries(monkeypatch):
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("Offline code touched a credential, microphone, or service boundary")

    for name in ("saved_gemini_key", "prompted_gemini_key", "save_gemini_key"):
        monkeypatch.setattr(cli, name, forbidden)
    for name in ("GeminiSceneDescriber", "_recognize_text", "_record_voice", "_transcribe_voice", "_parse_voice_command"):
        monkeypatch.setattr(phase4, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(phase4, "_Task", ImmediateTask)
    monkeypatch.setattr(phase4, "time", SimpleNamespace(monotonic=lambda: 10., monotonic_ns=lambda: 10_000_000_000,
                                                       sleep=lambda _seconds: None))
    yield
    assert not calls


def settings(tmp_path):
    return AppSettings(camera=CameraSettings(source="sample", width=160, height=120, fps=30),
                       simulation=SimulationSettings(preview=True, report_path=tmp_path / "offline.json"),
                       models=ModelSettings(scene_description_backend="gemini", ocr_backend="gemini",
                                            speech_recognition_backend="gemini"),
                       # A configured audio path must remain unopened while offline.
                       voice=VoiceSettings(audio_path=tmp_path / "missing-command.wav"))


def keys(monkeypatch, sequence):
    values = iter(sequence)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(values))


def test_offline_main_bypasses_all_credential_helpers_even_with_environment_key(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "OFFLINE-FAKE-KEY")
    configured = settings(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda _: configured)
    run = Mock(return_value=SimpleNamespace(interrupted=False))
    monkeypatch.setattr(controller, "run_simulation", run)
    assert cli.main(["--offline", "--scene-live", "--continuous", "--preview"]) == 0
    assert run.call_args.kwargs == dict(continuous=True, scene_live=True, proximity=False,
                                       speak=False, question="What is in front of me?", offline=True)
    assert "voice_commands" not in run.call_args.kwargs


@pytest.mark.parametrize("flags", [
    [], ["--scene-live"], ["--continuous"], ["--scene-live", "--continuous"],
    ["--scene-live", "--preview"],
    ["--scene-live", "--continuous", "--preview", "--voice-commands"],
    ["--scene-live", "--continuous", "--preview", "--prompt-key"],
    ["--scene-live", "--continuous", "--preview", "--describe"],
    ["--scene-live", "--continuous", "--preview", "--gemini-read"],
    ["--scene-live", "--continuous", "--preview", "--gemini-detect"],
    ["--scene-live", "--continuous", "--preview", "--gemini-live"],
])
def test_invalid_offline_cli_fails_before_authentication_or_controller(monkeypatch, flags):
    run = Mock()
    monkeypatch.setattr(controller, "run_simulation", run)
    assert cli.main(["--offline"] + flags) == 1
    run.assert_not_called()


@pytest.mark.parametrize("conflict", ["--prompt-key", "--voice-commands"])
def test_offline_auth_and_voice_conflict_rejected_before_even_loading_settings(monkeypatch, conflict):
    monkeypatch.setattr(cli, "load_settings", lambda _: pytest.fail("Settings loaded before explicit conflict guard"))
    assert cli.main(["--offline", conflict]) == 1


def test_direct_controller_validation_rejects_offline_without_scene_before_camera(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "_open_source", lambda _: pytest.fail("Camera opened"))
    with pytest.raises(ConfigurationError, match="offline requires"):
        controller.run_simulation(settings(tmp_path), offline=True)
    with pytest.raises(ConfigurationError, match="voice commands"):
        controller.run_simulation(settings(tmp_path), offline=True, continuous=True, scene_live=True, voice_commands=True)


def test_direct_controller_forwards_only_enabled_offline_option(monkeypatch, tmp_path):
    expected = object()
    run = Mock(return_value=expected)
    monkeypatch.setattr(phase4, "run_phase4_live", run)
    monkeypatch.setenv("DISPLAY", ":offline-test")
    assert controller.run_simulation(settings(tmp_path), offline=True, continuous=True, scene_live=True) is expected
    assert run.call_args.kwargs["offline"] is True
    assert "voice_commands" not in run.call_args.kwargs


def test_direct_live_call_cannot_combine_offline_and_voice(monkeypatch, tmp_path):
    with pytest.raises(controller.SimulationError, match="Offline mode cannot"):
        phase4.run_phase4_live(settings(tmp_path), offline=True, voice_commands=True,
                              proximity=False, speak=False, open_source=lambda _: pytest.fail("Camera opened"))


def test_repeated_upper_lower_cloud_keys_are_disabled_before_creating_any_task(monkeypatch, tmp_path):
    task = Mock(side_effect=AssertionError("A task was created"))
    monkeypatch.setattr(phase4, "_Task", task)
    pressed = [ord(letter) for letter in "dDtTmMsRdTm"] + [ord("q")]
    keys(monkeypatch, pressed)
    camera = Camera()
    result = phase4.run_phase4_live(settings(tmp_path), offline=True, proximity=False, speak=False,
                                   open_source=lambda _: camera)
    task.assert_not_called()
    report = json.loads(result.report_path.read_text())
    disabled = [item for item in report["actions"] if item["status"] == "disabled_offline"]
    assert len(disabled) == 9 and report["action_totals"]["cloud_requests"] == 0
    assert report["offline"] is True and report["cloud_uploads_disabled"] is True
    assert report["description"]["status"] == "not_requested" and camera.closed


def test_offline_h_and_p_use_local_fixture_and_explicit_speech(monkeypatch, tmp_path):
    fixture = tmp_path / "pair.json"
    fixture.write_text(json.dumps({"v": 1, "snapshots": [{
        "left": {"status": "ok", "distance_mm": 500},
        "right": {"status": "no_echo", "distance_mm": None},
    }]}))
    configured = replace(settings(tmp_path), proximity=ProximitySettings(backend="fixture", fixture_path=fixture))
    spoken = Mock(side_effect=lambda text, **kwargs: SpeechResult(status="completed", text=text, backend="fake-local"))
    monkeypatch.setattr(phase4, "LocalSpeech", Mock(return_value=SimpleNamespace(speak=spoken)))
    keys(monkeypatch, [ord("h"), -1, ord("p"), -1, ord("s"), ord("r"), ord("q")])
    result = phase4.run_phase4_live(configured, offline=True, proximity=True, speak=True,
                                   open_source=lambda _: Camera())
    assert spoken.call_count == 2
    assert spoken.call_args_list[0].args[0] == phase4.OFFLINE_HELP_TEXT
    assert "Simulated proximity" in spoken.call_args_list[1].args[0]
    report = json.loads(result.report_path.read_text())
    assert report["proximity"]["left"]["simulated"] is True
    assert report["action_totals"]["local_reads"] == 1
    assert report["action_totals"]["cloud_requests"] == 0


def test_offline_help_does_not_activate_speech_unless_requested(monkeypatch, tmp_path):
    local_speech = Mock(side_effect=AssertionError("Speech constructed without --speak"))
    monkeypatch.setattr(phase4, "LocalSpeech", local_speech)
    keys(monkeypatch, [ord("h"), ord("q")])
    result = phase4.run_phase4_live(settings(tmp_path), offline=True, proximity=False, speak=False,
                                   open_source=lambda _: Camera())
    local_speech.assert_not_called()
    assert result.speech.status == "not_requested"


def test_offline_preview_has_no_cloud_or_microphone_activation_prompt(monkeypatch):
    drawn = []
    monkeypatch.setattr(cv2, "putText", lambda image, text, *_args: drawn.append(text))
    packet = FramePacket(1, 10_000_000_000, np.zeros((240, 320, 3), np.uint8))
    phase4.render_phase4_preview(packet, None, {"status": "not_requested"},
                                 proximity_text="Simulated proximity.", cloud_busy=False,
                                 audio_busy=False, paused=False, offline=True)
    text = " ".join(drawn)
    assert "OFFLINE" in text and "Press H" in text
    assert "Press D" not in text and "M command" not in text and "D describe" not in text


def test_runtime_offline_guard_is_valid_without_changing_disabled_defaults(tmp_path):
    configured = settings(tmp_path)
    validate_runtime_settings(configured, offline=True, scene_live=True, continuous=True)
    assert configured.proximity.backend == "disabled"
    assert configured.voice.audio_path.name == "missing-command.wav"
