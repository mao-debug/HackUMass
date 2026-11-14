"""Voice auditions use a fixed sample and never capture images or call APIs."""

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
