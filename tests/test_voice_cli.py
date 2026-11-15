"""Voice CLI activation and routing never access real microphones or services."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vidi.main as cli
from vidi.core import controller, phase4
from vidi.core.configuration import AppSettings, ModelSettings, SimulationSettings, VoiceSettings


SCENE_FLAGS = ["--scene-live", "--continuous", "--preview"]


@pytest.fixture
def route(monkeypatch):
    run = Mock(return_value=SimpleNamespace(interrupted=False))
    monkeypatch.setattr(controller, "run_simulation", run)
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: nullcontext())
    return run


def test_explicit_voice_flag_reaches_controller_without_recording_or_uploading(route):
    assert cli.main(SCENE_FLAGS + ["--voice-commands"]) == 0
    assert route.call_args.kwargs == {
        "continuous": True, "scene_live": True, "proximity": False,
        "speak": False, "question": "What is in front of me?", "voice_commands": True,
    }


def test_existing_scene_mode_does_not_pass_voice_activation(route):
    assert cli.main(SCENE_FLAGS) == 0
    assert "voice_commands" not in route.call_args.kwargs


def test_audio_fixture_cli_path_uses_working_directory_and_remains_unopened(route, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main(SCENE_FLAGS + ["--voice-commands", "--command-audio", "missing.wav"]) == 0
    assert route.call_args.args[0].voice.audio_path == tmp_path / "missing.wav"
    assert not (tmp_path / "missing.wav").exists()


def test_microphone_override_is_one_literal_setting(route):
    device = "plughw:CARD=USB,DEV=0"
    assert cli.main(SCENE_FLAGS + ["--voice-commands", "--microphone-device", device]) == 0
    assert route.call_args.args[0].voice.device == device


@pytest.mark.parametrize("arguments", [
    ["--voice-commands"], ["--voice-commands", "--continuous", "--preview"],
    ["--voice-commands", "--scene-live", "--continuous"],
    ["--command-audio", "clip.wav"], ["--microphone-device", "pulse"],
    SCENE_FLAGS + ["--command-audio", "clip.wav"],
    SCENE_FLAGS + ["--voice-commands", "--command-audio", "clip.wav", "--microphone-device", "pulse"],
    SCENE_FLAGS + ["--voice-commands", "--microphone-device", "bad\ninput"],
    SCENE_FLAGS + ["--voice-commands", "--command-audio", "clip.mp3"],
    SCENE_FLAGS + ["--voice-commands", "--command-audio", "clip.wav", "--report", "clip.wav"],
])
def test_invalid_voice_flags_fail_before_key_lookup_or_controller(route, monkeypatch, arguments):
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: pytest.fail("Credentials accessed"))
    assert cli.main(arguments) == 1
    route.assert_not_called()


def test_device_override_does_not_silently_keep_configured_audio_fixture(route, monkeypatch, tmp_path):
    settings = AppSettings(voice=VoiceSettings(audio_path=tmp_path / "configured.wav"))
    monkeypatch.setattr(cli, "load_settings", lambda _: settings)
    assert cli.main(SCENE_FLAGS + ["--voice-commands", "--microphone-device", "pulse"]) == 1
    route.assert_not_called()


@pytest.mark.parametrize("field,value", [("speech_recognition_backend", "local"), ("ocr_backend", "tesseract")])
def test_unsupported_voice_backends_fail_before_controller(route, monkeypatch, field, value):
    monkeypatch.setattr(cli, "load_settings", lambda _: AppSettings(models=ModelSettings(**{field: value})))
    assert cli.main(SCENE_FLAGS + ["--voice-commands"]) == 1
    route.assert_not_called()


def test_voice_configuration_without_flag_is_not_an_activation(route, monkeypatch, tmp_path):
    settings = AppSettings(voice=VoiceSettings(device="nonexistent", audio_path=tmp_path / "missing.wav"))
    monkeypatch.setattr(cli, "load_settings", lambda _: settings)
    assert cli.main(SCENE_FLAGS) == 0
    assert "voice_commands" not in route.call_args.kwargs


def test_prompted_key_remains_allowed_with_explicit_voice_scene_mode(route, monkeypatch):
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(cli, "prompted_gemini_key", context)
    assert cli.main(SCENE_FLAGS + ["--voice-commands", "--prompt-key"]) == 0
    context.assert_called_once_with()


@pytest.mark.parametrize("enabled", [False, True])
def test_controller_forwards_voice_only_when_requested(monkeypatch, enabled):
    settings = AppSettings(simulation=SimulationSettings(preview=True))
    result = SimpleNamespace(interrupted=False)
    live = Mock(return_value=result)
    monkeypatch.setattr(phase4, "run_phase4_live", live)
    monkeypatch.setenv("DISPLAY", ":offline-test")
    assert controller.run_simulation(settings, continuous=True, scene_live=True, voice_commands=enabled) is result
    if enabled:
        assert live.call_args.kwargs["voice_commands"] is True
    else:
        assert "voice_commands" not in live.call_args.kwargs


def test_direct_controller_voice_without_scene_activation_fails_before_camera(monkeypatch):
    monkeypatch.setattr(controller, "_open_source", lambda _: pytest.fail("Camera opened"))
    with pytest.raises(ValueError, match="voice-commands requires"):
        controller.run_simulation(AppSettings(), voice_commands=True)


def test_help_explains_trigger_duration_and_cloud_audio_without_activation(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "vidi Phase 5" in help_text
    assert "--voice-commands" in help_text and "--command-audio" in help_text
    assert "5 seconds" in help_text and "audio to Gemini" in help_text
