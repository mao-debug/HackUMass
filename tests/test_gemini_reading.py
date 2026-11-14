"""Explicit one-snapshot Gemini reading, with all external boundaries mocked."""

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vidi.ai import gemini_text_recognition, text_recognition
from vidi.audio import text_to_speech
from vidi.core import controller
from vidi.core.configuration import load_settings, validate_runtime_settings
from vidi.core.types import FramePacket, OcrResult, SpeechResult

cli = importlib.import_module("vidi.main")
ROOT = Path(__file__).resolve().parents[1]
MODEL = "test-gemini-reading-model"
TEXT = "ROOM 201\nWelcome to vidi"


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Gemini reading integration tests must never access the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")


@pytest.fixture
def engines(monkeypatch):
    cloud = MagicMock()
    cloud.recognize.side_effect = lambda packet: OcrResult(
        status="ready", frame_id=packet.frame_id, backend="gemini", model=MODEL,
        text=TEXT, elapsed_s=0.25,
    )
    cloud_factory = MagicMock(return_value=cloud)
    local = MagicMock()
    local.recognize.side_effect = lambda packet: OcrResult(
        status="ready", frame_id=packet.frame_id, backend="tesseract", text="LOCAL TEXT",
    )
    local_factory = MagicMock(return_value=local)
    speaker = MagicMock()
    speaker.speak.return_value = SpeechResult(status="completed", mode="playback", text=TEXT)
    speech_factory = MagicMock(return_value=speaker)
    monkeypatch.setattr(gemini_text_recognition, "GeminiTextRecognizer", cloud_factory)
    monkeypatch.setattr(text_recognition, "TesseractRecognizer", local_factory)
    monkeypatch.setattr(text_to_speech, "LocalSpeech", speech_factory)
    return SimpleNamespace(
        cloud=cloud, cloud_factory=cloud_factory, local=local,
        local_factory=local_factory, speaker=speaker, speech_factory=speech_factory,
    )


def test_final_raw_camera_packet_is_read_once_after_close_with_checkpoints_before_speech(monkeypatch, tmp_path, engines, capsys):
    frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    packets = [FramePacket(101 + index, 1_000_000_000 + index, frame) for index, frame in enumerate(frames)]
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.side_effect = packets
    manager.__exit__.return_value = False
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    report, output = tmp_path / "reading.json", tmp_path / "snapshot.png"

    def recognize(packet):
        manager.__exit__.assert_called_once()
        assert packet is packets[-1]
        np.testing.assert_array_equal(packet.image_bgr, frames[-1])
        np.testing.assert_array_equal(cv2.imread(str(output)), frames[-1])
        checkpoint = json.loads(report.read_text())
        assert checkpoint["ocr"]["status"] == "pending"
        assert checkpoint["ocr"]["backend"] == "gemini"
        assert checkpoint["ocr"]["model"] == "gemini-3.5-flash-lite"
        assert checkpoint["speech"]["status"] == "not_requested"
        return OcrResult(status="ready", frame_id=packet.frame_id, backend="gemini", model=MODEL, text=TEXT)

    def speak(text, *, output_path):
        manager.__exit__.assert_called_once()
        checkpoint = json.loads(report.read_text())
        assert checkpoint["ocr"]["status"] == "ready"
        assert checkpoint["ocr"]["text"] == text == TEXT
        assert checkpoint["ocr"]["model"] == MODEL
        assert checkpoint["speech"]["status"] == "pending"
        assert output_path is None
        return SpeechResult(status="completed", mode="playback", text=text)

    engines.cloud.recognize.side_effect = recognize
    engines.speaker.speak.side_effect = speak
    assert cli.main([
        "--source", "picamera2", "--gemini-read", "--max-frames", "3", "--speak",
        "--output", str(output), "--report", str(report),
    ]) == 0
    engines.cloud.recognize.assert_called_once()
    engines.speaker.speak.assert_called_once_with(TEXT, output_path=None)
    engines.local_factory.assert_not_called()
    manager.__enter__.return_value.read.assert_not_called()
    data = json.loads(report.read_text())
    assert data["frames_processed"] == 3 and data["ocr"]["frame_id"] == 103
    assert data["ocr"]["text"] == TEXT and data["speech"]["status"] == "completed"
    assert data["ocr"]["model"] == MODEL and data["backend"] == "none"
    assert capsys.readouterr().out.count(TEXT) == 1


def test_config_and_key_without_action_never_construct_cloud_reader_or_access_key_context(monkeypatch, engines):
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.return_value = FramePacket(
        1, 1, np.zeros((48, 64, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    for name in ("saved_gemini_key", "prompted_gemini_key"):
        monkeypatch.setattr(cli, name, lambda: pytest.fail("Credential context entered without an upload action"))
    assert cli.main(["--config", str(ROOT / "config/pi-gemini-ocr.yaml")]) == 0
    engines.cloud_factory.assert_not_called()
    engines.local_factory.assert_not_called()
    engines.speech_factory.assert_not_called()


@pytest.mark.parametrize("extra", [
    ["--read-text"], ["--gemini-detect"], ["--gemini-live", "--continuous"],
    ["--describe"], ["--continuous"], ["--detector", "opencv_ssd"],
    ["--source", "imx500"],
])
def test_conflicting_actions_fail_before_camera_credentials_or_upload(monkeypatch, engines, extra):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened for invalid options"))
    for name in ("saved_gemini_key", "prompted_gemini_key"):
        monkeypatch.setattr(cli, name, lambda: pytest.fail("Credential access preceded option validation"))
    assert cli.main(["--gemini-read", "--prompt-key", *extra]) == 1
    engines.cloud_factory.assert_not_called()


def test_local_read_flag_cannot_silently_select_configured_gemini_backend(monkeypatch, engines):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: pytest.fail("Saved key accessed"))
    assert cli.main(["--config", str(ROOT / "config/pi-gemini-ocr.yaml"), "--read-text"]) == 1
    engines.cloud_factory.assert_not_called()
    engines.local_factory.assert_not_called()


def test_empty_gemini_read_uses_normal_feedback_once(monkeypatch, engines, capsys):
    engines.cloud.recognize.side_effect = lambda packet: OcrResult(
        status="no_text", frame_id=packet.frame_id, backend="gemini", model=MODEL,
    )
    assert cli.main(["--gemini-read", "--speak"]) == 0
    engines.speaker.speak.assert_called_once_with("", output_path=None)
    engines.local_factory.assert_not_called()
    assert capsys.readouterr().out == "No readable text found.\n"


def test_network_failure_preserves_raw_image_and_failed_report_without_fallback(engines, tmp_path, capsys):
    engines.cloud.recognize.side_effect = lambda packet: OcrResult(
        status="failed", frame_id=packet.frame_id, backend="gemini", model=MODEL,
        error="Network unavailable", elapsed_s=0.5,
    )
    output, report = tmp_path / "raw.png", tmp_path / "failed.json"
    assert cli.main(["--gemini-read", "--speak", "--output", str(output), "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    assert output.is_file()
    assert data["ocr"]["status"] == "failed" and data["ocr"]["text"] == ""
    assert data["ocr"]["model"] == MODEL and data["ocr"]["error"] == "Network unavailable"
    assert data["speech"]["status"] == "skipped_ocr_failed"
    assert capsys.readouterr().out == ""
    engines.speech_factory.assert_not_called()
    engines.local_factory.assert_not_called()


@pytest.mark.parametrize("stage", ["ocr", "speech"])
def test_cancellation_retains_backend_model_and_completed_reading(engines, tmp_path, stage):
    if stage == "ocr":
        engines.cloud.recognize.side_effect = KeyboardInterrupt
    else:
        engines.speaker.speak.side_effect = KeyboardInterrupt
    report = tmp_path / "cancelled.json"
    assert cli.main(["--gemini-read", "--speak", "--report", str(report)]) == 130
    data = json.loads(report.read_text())
    assert data[stage]["status"] == "cancelled"
    assert data["ocr"]["backend"] == "gemini"
    if stage == "ocr":
        assert data["ocr"]["model"] == "gemini-3.5-flash-lite"
        assert data["ocr"]["text"] == ""
        engines.speech_factory.assert_not_called()
    else:
        assert data["ocr"]["status"] == "ready" and data["ocr"]["text"] == TEXT
        assert data["ocr"]["model"] == MODEL
    engines.local_factory.assert_not_called()


def test_failed_speech_retains_full_gemini_transcription_and_model(engines, tmp_path):
    engines.speaker.speak.return_value = SpeechResult(status="failed", mode="playback", error="Audio unavailable")
    report = tmp_path / "audio-failed.json"
    assert cli.main(["--gemini-read", "--speak", "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["ocr"]["status"] == "ready" and data["ocr"]["text"] == TEXT
    assert data["ocr"]["model"] == MODEL and data["speech"]["status"] == "failed"
    engines.cloud.recognize.assert_called_once()


def test_failed_pending_report_prevents_upload_or_speech(engines, tmp_path):
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("file", encoding="utf-8")
    assert cli.main(["--gemini-read", "--speak", "--report", str(parent_file / "report.json")]) == 1
    engines.cloud.recognize.assert_not_called()
    engines.speech_factory.assert_not_called()


def test_gemini_read_can_export_speech_without_requesting_playback(engines, tmp_path):
    destination, report = tmp_path / "reading.wav", tmp_path / "reading.json"
    engines.speaker.speak.return_value = SpeechResult(
        status="completed", mode="wav", backend="espeak_ng", text=TEXT, output_path=destination,
    )
    assert cli.main([
        "--gemini-read", "--speech-output", str(destination), "--report", str(report),
    ]) == 0
    engines.speaker.speak.assert_called_once_with(TEXT, output_path=destination)
    data = json.loads(report.read_text())
    assert data["ocr"]["text"] == TEXT and data["ocr"]["model"] == MODEL
    assert data["speech"]["mode"] == "wav" and data["speech"]["output_path"] == str(destination)


@pytest.mark.parametrize("prompted", [False, True])
def test_new_action_uses_selected_saved_or_prompted_key_context(monkeypatch, engines, prompted):
    events = []

    @contextmanager
    def key_context():
        events.append("entered")
        yield None
        events.append("closed")

    selected = "prompted_gemini_key" if prompted else "saved_gemini_key"
    other = "saved_gemini_key" if prompted else "prompted_gemini_key"
    monkeypatch.setattr(cli, selected, key_context)
    monkeypatch.setattr(cli, other, lambda: pytest.fail("Wrong credential context selected"))
    extra = ["--prompt-key"] if prompted else []
    assert cli.main(["--gemini-read", *extra]) == 0
    assert events == ["entered", "closed"]
    engines.cloud.recognize.assert_called_once()


def test_existing_local_reading_stays_local_and_does_not_load_gemini(engines):
    assert cli.main(["--read-text"]) == 0
    engines.local.recognize.assert_called_once()
    engines.cloud_factory.assert_not_called()
    engines.speech_factory.assert_not_called()


def test_gemini_reading_profile_selects_one_snapshot_with_separate_ocr_timeout():
    settings = load_settings(ROOT / "config/pi-gemini-ocr.yaml")
    validate_runtime_settings(settings, gemini_read=True, speak=True)
    assert settings.models.object_detection_backend == "none"
    assert settings.models.ocr_backend == "gemini"
    assert settings.models.object_detection_path is None
    assert settings.camera.source == "picamera2" and settings.simulation.max_frames == 1
    assert (settings.camera.width, settings.camera.height) == (1280, 960)
    assert settings.ocr.timeout_s == 60.0
