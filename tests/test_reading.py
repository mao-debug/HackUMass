"""Reading workflow contracts without camera, OCR engine, or audio access."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vidi.ai import text_recognition
from vidi.audio import text_to_speech
from vidi.core import controller
from vidi.core.configuration import (
    AppSettings, ConfigurationError, ModelSettings, SimulationSettings, load_settings,
    validate_runtime_settings,
)
from vidi.core.types import OcrResult, SpeechResult
from vidi.main import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_engines(monkeypatch):
    recognizer = MagicMock()
    recognizer.recognize.return_value = OcrResult(
        status="ready", frame_id=1, text="ROOM 201\nWelcome to vidi", backend="tesseract", elapsed_s=0.1,
    )
    speaker = MagicMock()
    speaker.speak.return_value = SpeechResult(
        status="completed", mode="playback", backend="say", text="ROOM 201 Welcome to vidi", elapsed_s=0.2,
    )
    ocr_factory = MagicMock(return_value=recognizer)
    speech_factory = MagicMock(return_value=speaker)
    monkeypatch.setattr(text_recognition, "TesseractRecognizer", ocr_factory)
    monkeypatch.setattr(text_to_speech, "LocalSpeech", speech_factory)
    return recognizer, speaker, ocr_factory, speech_factory


def test_configured_engines_and_api_key_never_enable_reading_or_audio(monkeypatch, local_engines):
    _, _, ocr_factory, speech_factory = local_engines
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    result = controller.run_simulation(AppSettings(models=ModelSettings(
        ocr_backend="tesseract", text_to_speech_backend="say", scene_description_backend="gemini",
    )))
    assert result.ocr.status == result.speech.status == "not_requested"
    ocr_factory.assert_not_called()
    speech_factory.assert_not_called()


def test_pi_reading_profile_has_bounded_camera_preparation_and_no_detector():
    settings = load_settings(ROOT / "config/pi-ocr.yaml")
    validate_runtime_settings(settings, read_text=True, speak=True)
    assert (settings.camera.width, settings.camera.height) == (1280, 960)
    assert settings.camera.source == "picamera2" and settings.simulation.max_frames == 1
    assert settings.models.object_detection_backend == "none"
    assert settings.camera.settle_s == 1.0 and settings.camera.autofocus == "single_if_supported"
    assert settings.ocr.language == "eng" and settings.ocr.psm == 6
    assert settings.speech.max_words == 60 and settings.speech.speed_wpm == 150


def test_raw_frame_survives_camera_close_and_ocr_is_checkpointed_before_speech(monkeypatch, tmp_path, local_engines, capsys):
    recognizer, speaker, _, _ = local_engines
    frame = np.full((48, 64, 3), (10, 20, 30), dtype=np.uint8)
    manager = MagicMock()
    manager.__enter__.return_value.read.return_value = frame
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    report = tmp_path / "reading.json"

    def recognize(packet):
        manager.__exit__.assert_called_once()
        np.testing.assert_array_equal(packet.image_bgr, frame)
        return OcrResult(status="ready", frame_id=packet.frame_id, text="ROOM 201", backend="tesseract")

    def speak(text, *, output_path):
        manager.__exit__.assert_called_once()
        checkpoint = json.loads(report.read_text())
        assert checkpoint["ocr"]["text"] == text == "ROOM 201"
        assert checkpoint["speech"]["status"] == "pending"
        assert output_path is None
        return SpeechResult(status="completed", mode="playback", backend="say", text=text)

    recognizer.recognize.side_effect = recognize
    speaker.speak.side_effect = speak
    settings = AppSettings(simulation=SimulationSettings(report_path=report))
    result = controller.run_simulation(settings, read_text=True, speak=True)
    assert result.ocr.text == "ROOM 201" and result.speech.status == "completed"
    assert json.loads(report.read_text())["speech"]["status"] == "completed"
    assert "ROOM 201" in capsys.readouterr().out


def test_reading_without_speech_prints_full_text_once(local_engines, capsys):
    _, _, _, speech_factory = local_engines
    result = controller.run_simulation(AppSettings(), read_text=True)
    assert capsys.readouterr().out == "ROOM 201\nWelcome to vidi\n"
    assert result.speech.status == "not_requested"
    speech_factory.assert_not_called()


def test_failed_ocr_preserves_report_and_never_starts_speech(local_engines, tmp_path):
    recognizer, _, _, speech_factory = local_engines
    recognizer.recognize.return_value = OcrResult(status="failed", backend="tesseract", error="Engine missing")
    report = tmp_path / "failure.json"
    assert main(["--read-text", "--speak", "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["ocr"]["status"] == "failed"
    assert data["speech"]["status"] == "skipped_ocr_failed"
    speech_factory.assert_not_called()


def test_empty_ocr_requests_one_brief_speech_feedback_and_succeeds(local_engines, capsys):
    recognizer, speaker, _, _ = local_engines
    recognizer.recognize.return_value = OcrResult(status="no_text", frame_id=1, backend="tesseract")
    assert main(["--read-text", "--speak"]) == 0
    speaker.speak.assert_called_once_with("", output_path=None)
    assert "No readable text found." in capsys.readouterr().out


def test_speech_failure_returns_nonzero_and_retains_full_ocr(local_engines, tmp_path, capsys):
    _, speaker, _, _ = local_engines
    speaker.speak.return_value = SpeechResult(status="failed", mode="playback", error="No engine")
    report = tmp_path / "speech-failure.json"
    assert main(["--read-text", "--speak", "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["ocr"]["text"] == "ROOM 201\nWelcome to vidi"
    assert data["speech"]["status"] == "failed"
    assert "ROOM 201" in capsys.readouterr().out


@pytest.mark.parametrize("stage", ["ocr", "speech"])
def test_interruption_retains_checkpoint_and_returns_130(stage, local_engines, tmp_path):
    recognizer, speaker, _, _ = local_engines
    if stage == "ocr":
        recognizer.recognize.side_effect = KeyboardInterrupt
    else:
        speaker.speak.side_effect = KeyboardInterrupt
    report = tmp_path / "cancelled.json"
    assert main(["--read-text", "--speak", "--report", str(report)]) == 130
    data = json.loads(report.read_text())
    assert data[stage]["status"] == "cancelled"
    if stage == "speech":
        assert data["ocr"]["text"] == "ROOM 201\nWelcome to vidi"


def test_failed_report_checkpoint_prevents_speech(local_engines, tmp_path):
    _, _, _, speech_factory = local_engines
    parent = tmp_path / "file"
    parent.write_text("not a directory")
    assert main(["--read-text", "--speak", "--report", str(parent / "report.json")]) == 1
    speech_factory.assert_not_called()


def test_failed_cancellation_checkpoint_does_not_mask_ctrl_c(monkeypatch, local_engines, tmp_path):
    _, speaker, _, _ = local_engines
    speaker.speak.side_effect = KeyboardInterrupt
    report = tmp_path / "cancelled.json"
    save_report = controller._save_report

    def checkpoint(path, data):
        if data["speech"]["status"] == "cancelled":
            raise controller.SimulationError("Filesystem unavailable")
        save_report(path, data)

    monkeypatch.setattr(controller, "_save_report", checkpoint)
    assert main(["--read-text", "--speak", "--report", str(report)]) == 130
    assert json.loads(report.read_text())["ocr"]["text"] == "ROOM 201\nWelcome to vidi"


def test_wav_path_and_mode_are_reported_without_playback(local_engines, tmp_path):
    _, speaker, _, _ = local_engines
    output, report = tmp_path / "reading.wav", tmp_path / "reading.json"
    speaker.speak.return_value = SpeechResult(status="completed", mode="wav", output_path=output)
    assert main(["--read-text", "--speech-output", str(output), "--report", str(report)]) == 0
    speaker.speak.assert_called_once_with("ROOM 201\nWelcome to vidi", output_path=output)
    data = json.loads(report.read_text())
    assert data["speech"]["mode"] == "wav" and data["speech"]["output_path"] == str(output)


@pytest.mark.parametrize("arguments", [
    ["--speak"], ["--speech-output", "read.wav"],
    ["--read-text", "--describe"], ["--read-text", "--detector", "fixture"],
    ["--read-text", "--speech-output", "read.mp3"],
    ["--read-text", "--output", "same.wav", "--speech-output", "same.wav"],
    ["--read-text", "--report", "same.wav", "--speech-output", "same.wav"],
    ["--read-text", "--image", "same.png", "--report", "same.png"],
    ["--read-text", "--image", "same.png", "--output", "same.png"],
])
def test_invalid_reading_options_fail_before_camera(monkeypatch, arguments):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main(arguments) == 1


def test_playback_and_export_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        main(["--read-text", "--speak", "--speech-output", "read.wav"])
    assert error.value.code == 2


@pytest.mark.parametrize("yaml", [
    "ocr: {timeout_s: .nan}", "ocr: {timeout_s: false}", "ocr: {psm: true}",
    "ocr: {psm: 0}", "ocr: {preprocess: unsupported}", "ocr: {language: '-l eng'}",
    "ocr: {language: []}", "ocr: {executable: null}",
    "speech: {max_words: 0}", "speech: {speed_wpm: false}", "speech: {timeout_s: .inf}",
    "speech: {voice: []}", "camera: {settle_s: false}", "camera: {settle_s: -1}",
    "camera: {autofocus: off}", "camera: {autofocus_timeout_s: 0}",
    "camera: {startup_timeout_s: 2, settle_s: 1, autofocus: single_if_supported}",
])
def test_invalid_reading_configuration_is_rejected(tmp_path, yaml):
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml)
    with pytest.raises(ConfigurationError):
        load_settings(config)


def test_atomic_report_failure_preserves_previous_checkpoint(monkeypatch, tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"ocr": "previous result"}\n')
    original = Path.replace

    def fail_replace(source, destination):
        if destination == path:
            raise OSError("Cannot publish")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(controller.SimulationError, match="Cannot save JSON"):
        controller._save_report(path, {"ocr": "new result"})
    assert json.loads(path.read_text()) == {"ocr": "previous result"}
    assert list(tmp_path.iterdir()) == [path]
