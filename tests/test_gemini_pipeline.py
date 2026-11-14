"""Explicit cloud snapshot actions using fakes, without service connections."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vidi.ai import gemini_detection, scene_description
from vidi.audio import text_to_speech
from vidi.core import controller
from vidi.core.configuration import load_settings
from vidi.core.types import Detection, DetectionResult, SpeechResult
from vidi.main import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def detector(monkeypatch):
    model = MagicMock()
    model.detect.side_effect = lambda packet: DetectionResult(
        packet.frame_id, "gemini", "ready", (Detection(None, "chair", None, (10, 15, 40, 45)),),
        model="test-model", elapsed_s=0.2,
    )
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(gemini_detection, "GeminiObjectDetector", factory)
    return model, factory


def test_backend_profile_or_key_never_enable_upload_without_flag(monkeypatch, detector):
    model, factory = detector
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main(["--config", str(ROOT / "config/pi-gemini.yaml")]) == 1
    assert main(["--detector", "gemini"]) == 1
    model.detect.assert_not_called()
    factory.assert_not_called()


def test_ordinary_run_does_not_construct_cloud_detector(detector):
    _, factory = detector
    assert main([]) == 0
    factory.assert_not_called()


def test_exactly_one_final_raw_snapshot_is_submitted_after_camera_closes(monkeypatch, tmp_path, detector, capsys):
    model, _ = detector
    frames = [np.full((48, 64, 3), n, dtype=np.uint8) for n in (10, 20, 30)]
    manager = MagicMock()
    manager.__enter__.return_value.read.side_effect = frames
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    report, output = tmp_path / "gemini.json", tmp_path / "gemini.png"

    def detect(packet):
        manager.__exit__.assert_called_once()
        np.testing.assert_array_equal(packet.image_bgr, frames[-1])
        assert packet.frame_id == 3
        assert json.loads(report.read_text())["last_inference"]["status"] == "pending"
        np.testing.assert_array_equal(cv2.imread(str(output)), frames[-1])
        return DetectionResult(packet.frame_id, "gemini", "ready", (
            Detection(None, "chair", None, (10, 15, 40, 45)),
        ), model="test-model", elapsed_s=0.2)

    model.detect.side_effect = detect
    assert main([
        "--gemini-detect", "--max-frames", "3", "--output", str(output), "--report", str(report),
    ]) == 0
    model.detect.assert_called_once()
    data = json.loads(report.read_text())
    assert data["frames_processed"] == 3 and data["ready_frames"] == 1
    assert data["inference_ready_fps"] is None
    assert data["frames"][0]["inference"] is None and data["frames"][1]["inference"] is None
    assert data["frames"][-1]["inference"]["frame_id"] == 3
    assert data["last_inference"]["model"] == "test-model"
    assert data["last_inference"]["elapsed_s"] == 0.2
    assert data["detections"][0]["class_id"] is None and data["detections"][0]["confidence"] is None
    assert data["backend"] == "gemini" and data["simulated"] is False
    assert not np.array_equal(cv2.imread(str(output)), frames[-1])
    assert "chair" in capsys.readouterr().out


def test_unavailable_cloud_preserves_artifacts_and_does_not_start_speech(monkeypatch, tmp_path, detector):
    model, _ = detector
    model.detect.side_effect = lambda packet: DetectionResult(
        packet.frame_id, "gemini", "unavailable", model="test-model", error="Network unavailable",
    )
    speaker = MagicMock()
    monkeypatch.setattr(text_to_speech, "LocalSpeech", speaker)
    output, report = tmp_path / "unavailable.png", tmp_path / "unavailable.json"
    assert main([
        "--gemini-detect", "--speak", "--output", str(output), "--report", str(report),
    ]) == 1
    data = json.loads(report.read_text())
    assert output.is_file()
    assert data["last_inference_status"] == "unavailable"
    assert data["last_inference"]["error"] == "Network unavailable"
    assert data["detections"] == [] and data["ready_frames"] == 0
    assert data["speech"]["status"] == "skipped_cloud_failed"
    speaker.assert_not_called()


def test_cloud_cancellation_preserves_snapshot_and_records_130(tmp_path, detector):
    model, _ = detector
    model.detect.side_effect = KeyboardInterrupt
    report, output = tmp_path / "cancelled.json", tmp_path / "raw.png"
    assert main(["--gemini-detect", "--report", str(report), "--output", str(output)]) == 130
    data = json.loads(report.read_text())
    assert data["last_inference_status"] == data["last_inference"]["status"] == "cancelled"
    assert data["detections"] == [] and output.is_file()


def test_ready_empty_cloud_result_has_specific_feedback(monkeypatch, detector):
    model, _ = detector
    model.detect.side_effect = lambda packet: DetectionResult(packet.frame_id, "gemini", "ready")
    speaker = MagicMock()
    speaker.speak.return_value = SpeechResult(status="completed", mode="playback")
    monkeypatch.setattr(text_to_speech, "LocalSpeech", lambda *args: speaker)
    assert main(["--gemini-detect", "--speak"]) == 0
    speaker.speak.assert_called_once_with("No objects identified in this snapshot.", output_path=None)


def test_failed_speech_preserves_completed_detections(monkeypatch, detector, tmp_path):
    speaker = MagicMock()
    speaker.speak.return_value = SpeechResult(status="failed", error="No audio engine")
    monkeypatch.setattr(text_to_speech, "LocalSpeech", lambda *args: speaker)
    report = tmp_path / "cloud-with-speech-error.json"
    assert main(["--gemini-detect", "--speak", "--report", str(report)]) == 1
    data = json.loads(report.read_text())
    assert data["last_inference"]["status"] == "ready" and data["detections"][0]["label"] == "chair"
    assert data["speech"]["status"] == "failed"


def test_gemini_description_can_now_be_spoken_without_ocr(monkeypatch, tmp_path, caplog):
    describer = MagicMock()
    describer.describe.return_value = scene_description.SceneDescriptionResult("A chair is visible.", "test-model", 0.2)
    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", lambda *args: describer)
    speaker = MagicMock()
    speaker.speak.return_value = SpeechResult(status="completed", mode="playback")
    monkeypatch.setattr(text_to_speech, "LocalSpeech", lambda *args: speaker)
    report = tmp_path / "description.json"
    assert main(["--describe", "--speak", "--report", str(report)]) == 0
    speaker.speak.assert_called_once_with("A chair is visible.", output_path=None)
    data = json.loads(report.read_text())
    assert data["ocr"]["status"] == "not_requested" and data["speech"]["status"] == "completed"
    assert "audio are not active" not in caplog.text


def test_preview_quit_finishes_capture_before_one_explicit_upload(monkeypatch, detector):
    model, _ = detector
    titles = []
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(controller.cv2, "imshow", lambda title, image: titles.append(title))
    monkeypatch.setattr(controller.cv2, "waitKey", lambda milliseconds: 27)
    monkeypatch.setattr(controller.cv2, "destroyAllWindows", lambda: None)
    assert main(["--gemini-detect", "--preview", "--max-frames", "3"]) == 0
    model.detect.assert_called_once()
    assert model.detect.call_args.args[0].frame_id == 1
    assert "finishes capture" in titles[0] and titles[-1] == "vidi Gemini result"


def test_description_cancellation_preserves_snapshot_and_report(monkeypatch, tmp_path):
    describer = MagicMock()
    report, output = tmp_path / "description.json", tmp_path / "raw.png"

    def describe(*args):
        assert json.loads(report.read_text())["description_status"] == "pending"
        raise KeyboardInterrupt

    describer.describe.side_effect = describe
    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", lambda *args: describer)
    assert main(["--describe", "--report", str(report), "--output", str(output)]) == 130
    assert output.is_file()
    assert json.loads(report.read_text())["description_status"] == "cancelled"


@pytest.mark.parametrize("arguments", [
    ["--gemini-detect", "--describe"], ["--gemini-detect", "--read-text"],
    ["--gemini-detect", "--detector", "opencv_ssd"],
    ["--gemini-detect", "--source", "imx500"], ["--gemini-detect", "--max-detections", "21"],
    ["--gemini-detect", "--labels", "labels.txt"],
    ["--gemini-detect", "--detections-fixture", "annotations.json"],
    ["--gemini-detect", "--image", "same.png", "--report", "same.png"],
    ["--describe", "--image", "same.png", "--output", "same.png"],
    ["--gemini-detect", "--threshold", "0.9"],
    ["--gemini-detect", "--model", "local.caffemodel"],
    ["--gemini-detect", "--model-config", "local.prototxt"],
])
def test_cloud_option_conflicts_fail_before_camera(monkeypatch, arguments):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main(arguments) == 1


def test_gemini_profile_prepares_one_camera_frame_without_local_model():
    settings = load_settings(ROOT / "config/pi-gemini.yaml")
    assert settings.camera.source == "picamera2"
    assert (settings.camera.width, settings.camera.height) == (1280, 960)
    assert settings.simulation.max_frames == 1
    assert settings.models.object_detection_backend == "gemini"
    assert settings.models.object_detection_path is None
    assert settings.cloud.detection_max_output_tokens == 1024


def test_mismatched_cloud_frame_is_not_accepted(detector, tmp_path):
    model, _ = detector
    model.detect.return_value = DetectionResult(99, "gemini", "ready")
    model.detect.side_effect = None
    assert main(["--gemini-detect", "--report", str(tmp_path / "mismatch.json")]) == 1
