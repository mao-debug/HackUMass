"""Voice configuration stays inert and bounds explicit recording requests."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from vidi.core.configuration import (
    AppSettings, ConfigurationError, ModelSettings, ProximitySettings,
    SimulationSettings, VoiceSettings, load_settings, validate_runtime_settings,
)


def active_settings(**changes):
    return replace(AppSettings(simulation=SimulationSettings(preview=True)), **changes)


def validate_voice(settings):
    validate_runtime_settings(settings, voice_commands=True, scene_live=True, continuous=True)


def test_voice_defaults_are_bounded_and_immutable_without_enabling_any_input():
    settings = load_settings()
    assert settings.voice == VoiceSettings(device="pulse", record_seconds=5, timeout_s=30.0)
    assert settings.voice.audio_path is None and settings.voice.model is None
    validate_runtime_settings(settings)
    with pytest.raises(FrozenInstanceError):
        settings.voice.record_seconds = 10


@pytest.mark.parametrize("values", [
    {"device": None}, {"device": ""}, {"device": " \t"}, {"device": 2},
    {"device": "pulse\x00other"}, {"device": "pulse\nother"}, {"device": "pulse\rother"},
    {"record_seconds": 0}, {"record_seconds": 6}, {"record_seconds": True}, {"record_seconds": 1.5},
    {"timeout_s": 0}, {"timeout_s": 60.1}, {"timeout_s": True},
    {"timeout_s": float("inf")}, {"timeout_s": float("nan")},
    {"model": ""}, {"model": "bad\x00name"}, {"audio_path": "clip.wav"},
])
def test_invalid_or_unbounded_voice_settings_are_rejected(values):
    with pytest.raises(ConfigurationError):
        VoiceSettings(**values)


def test_boundary_values_and_literal_alsa_device_are_preserved():
    settings = VoiceSettings(device="plughw:CARD=USB,DEV=0", record_seconds=1, timeout_s=60)
    assert settings.device == "plughw:CARD=USB,DEV=0"
    assert settings.record_seconds == 1


def test_yaml_voice_path_is_relative_to_config_without_reading_audio_or_importing_services(tmp_path, monkeypatch):
    import builtins

    config = tmp_path / "settings.yaml"
    config.write_text("voice:\n  device: pulse\n  audio_path: clips/missing.wav\n"
                      "  model: chosen-model\n  record_seconds: 3\n")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"sounddevice", "google", "picamera2", "libcamera", "serial"}
        assert name != "vidi.audio.speech_to_text"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    settings = load_settings(config)
    assert settings.voice.audio_path == tmp_path / "clips" / "missing.wav"
    assert not settings.voice.audio_path.exists()
    assert settings.voice.model == "chosen-model"
    assert settings.voice.record_seconds == 3
    # Configuration alone does not require voice mode, microphone, or a key.
    validate_runtime_settings(settings)


@pytest.mark.parametrize("document", [
    "voice: [pulse]\n", "voice: {automatic_listening: true}\n",
    "voice: {audio_path: 3}\n", "voice: {record_seconds: 6}\n",
])
def test_yaml_voice_schema_rejects_unknown_options_and_wrong_types(tmp_path, document):
    config = tmp_path / "invalid.yaml"
    config.write_text(document)
    with pytest.raises(ConfigurationError):
        load_settings(config)


@pytest.mark.parametrize("scene_live,continuous,preview", [
    (False, False, False), (False, True, True), (True, False, True), (True, True, False),
])
def test_voice_mode_requires_explicit_scene_live_continuous_preview(scene_live, continuous, preview):
    settings = AppSettings(simulation=SimulationSettings(preview=preview))
    with pytest.raises(ConfigurationError, match="voice-commands requires"):
        validate_runtime_settings(settings, voice_commands=True, scene_live=scene_live, continuous=continuous)


@pytest.mark.parametrize("field,value", [
    ("speech_recognition_backend", "vosk"), ("speech_recognition_backend", "local"),
    ("ocr_backend", "tesseract"),
])
def test_voice_uses_only_selected_supported_recognition_and_reading_backends(field, value):
    settings = active_settings(models=ModelSettings(**{field: value}))
    validate_runtime_settings(settings)  # Inert settings retain backward compatibility.
    with pytest.raises(ConfigurationError, match="gemini"):
        validate_voice(settings)


@pytest.mark.parametrize("backend", [None, "gemini"])
def test_voice_backend_fallback_and_explicit_gemini_are_valid(backend):
    settings = active_settings(models=ModelSettings(speech_recognition_backend=backend, ocr_backend=backend))
    validate_voice(settings)


def test_active_audio_fixture_requires_wav_but_configuration_does_not_read_it(tmp_path):
    settings = active_settings(voice=VoiceSettings(audio_path=tmp_path / "not-audio.txt"))
    validate_runtime_settings(settings)
    with pytest.raises(ConfigurationError, match="local .wav input"):
        validate_voice(settings)


@pytest.mark.parametrize("target", ["output_path", "report_path"])
def test_voice_exports_cannot_overwrite_selected_audio_input(tmp_path, target):
    audio = tmp_path / "command.wav"
    audio.write_bytes(b"preserve this input")
    settings = active_settings(
        voice=VoiceSettings(audio_path=audio),
        simulation=SimulationSettings(preview=True, **{target: audio}),
    )
    with pytest.raises(ConfigurationError, match="voice audio input"):
        validate_voice(settings)
    assert audio.read_bytes() == b"preserve this input"


def test_existing_sensor_fixture_export_protection_applies_with_voice_commands(tmp_path):
    fixture = tmp_path / "sensors.json"
    settings = active_settings(
        proximity=ProximitySettings(backend="fixture", fixture_path=fixture),
        simulation=SimulationSettings(preview=True, report_path=fixture),
    )
    with pytest.raises(ConfigurationError, match="proximity fixture"):
        validate_runtime_settings(settings, voice_commands=True, scene_live=True, continuous=True, proximity=True)
