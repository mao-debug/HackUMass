"""Fixed voice auditions keep local runs offline and gate cloud selection."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import socket
from unittest.mock import Mock

import pytest

from vidi.core.types import SpeechResult


spec = importlib.util.spec_from_file_location(
    "preview_voice", Path(__file__).resolve().parents[1] / "scripts" / "preview_voice.py",
)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


@pytest.fixture
def speaker(monkeypatch):
    engine = Mock()
    engine.speak.return_value = SpeechResult(status="completed", backend="piper", mode="playback")
    factory = Mock(return_value=engine)
    monkeypatch.setattr(preview, "LocalSpeech", factory)
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: pytest.fail("Network used during local audition"))
    return engine, factory


def test_piper_audition_uses_only_speech_settings_without_opening_camera(speaker):
    engine, factory = speaker
    config = Path(__file__).resolve().parents[1] / "config/pi-piper-ocr.yaml"
    assert preview.main(["--config", str(config)]) == 0
    assert factory.call_args.args[1] == "piper"
    engine.speak.assert_called_once_with(preview.SAMPLE_TEXT, output_path=None)
    assert "voice test" in preview.SAMPLE_TEXT


def test_audition_can_export_without_playback(tmp_path, speaker):
    engine, _ = speaker
    output = tmp_path / "voice.wav"
    assert preview.main(["--output", str(output)]) == 0
    engine.speak.assert_called_once_with(preview.SAMPLE_TEXT, output_path=output)


def test_audition_failure_is_reported(speaker, caplog):
    engine, _ = speaker
    engine.speak.return_value = SpeechResult(status="failed", error="Voice model missing")
    assert preview.main([]) == 1
    assert "Voice model missing" in caplog.text


def test_audition_cancellation_returns_130(speaker):
    engine, _ = speaker
    engine.speak.side_effect = KeyboardInterrupt
    assert preview.main([]) == 130


def test_audition_rejects_non_wav_export_before_engine(tmp_path, speaker):
    engine, factory = speaker
    assert preview.main(["--output", str(tmp_path / "recording.mp3")]) == 1
    engine.speak.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("arguments", [
    ["--backend", "gemini"],
    ["--backend", "piper", "--allow-cloud-speech"],
    ["--backend", "say", "--gemini-voice", "Kore"],
    ["--backend", "say", "--piper-model", "existing.onnx"],
    ["--repeat", "0"],
    ["--repeat", "6"],
    ["--repeat", "2", "--output", "sample.wav"],
])
def test_invalid_cloud_and_backend_combinations_fail_before_key_or_engine(arguments, speaker, monkeypatch):
    from vidi.audio import speech_factory
    from vidi.core import credentials

    _, local = speaker
    factory = Mock(side_effect=AssertionError("Speech factory reached"))
    key = Mock(side_effect=AssertionError("Key storage accessed"))
    monkeypatch.setattr(speech_factory, "create_speech", factory)
    monkeypatch.setattr(credentials, "saved_gemini_key", key)
    assert preview.main(arguments) == 1
    local.assert_not_called()
    factory.assert_not_called()
    key.assert_not_called()


def test_explicit_cloud_audition_uses_only_fixed_chess_sample_and_closes(speaker, monkeypatch):
    from vidi.audio import speech_factory
    from vidi.core import credentials

    _, local = speaker
    engine = Mock()
    engine.speak.return_value = SpeechResult(status="completed", backend="gemini")
    factory = Mock(return_value=engine)
    active = []

    @contextmanager
    def selected_key():
        active.append("opened")
        yield
        active.append("closed")

    monkeypatch.setattr(speech_factory, "create_speech", factory)
    monkeypatch.setattr(credentials, "saved_gemini_key", selected_key)
    assert preview.main(["--backend", "gemini", "--allow-cloud-speech", "--gemini-voice", "Kore", "--sample", "chess"]) == 0
    assert active == ["opened", "closed"]
    assert factory.call_args.args[1] == "gemini"
    assert factory.call_args.args[0].gemini_voice == "Kore"
    assert factory.call_args.kwargs == {"cloud_enabled": True}
    spoken = engine.speak.call_args.args[0]
    assert spoken == speech_factory.render_chess_speech(preview.CHESS_SAMPLE_TEXT)
    assert "G one to F three" in spoken
    engine.close.assert_called_once()
    local.assert_not_called()


def test_repeated_piper_reuses_one_session_without_key_access(speaker, monkeypatch, tmp_path):
    from vidi.audio import speech_factory
    from vidi.core import credentials

    _, local = speaker
    engine = Mock()
    engine.speak.return_value = SpeechResult(status="completed", backend="piper")
    factory = Mock(return_value=engine)
    key = Mock(side_effect=AssertionError("Local preview accessed key storage"))
    monkeypatch.setattr(speech_factory, "create_speech", factory)
    monkeypatch.setattr(credentials, "saved_gemini_key", key)
    runtime = tmp_path / "speech runtime" / "bin" / "python"
    assert preview.main(["--backend", "piper", "--repeat", "2", "--piper-python", str(runtime)]) == 0
    factory.assert_called_once()
    assert factory.call_args.args[0].piper_python_path == runtime
    assert factory.call_args.kwargs == {"cloud_enabled": False}
    assert engine.speak.call_count == 2
    engine.close.assert_called_once()
    local.assert_not_called()
    key.assert_not_called()
