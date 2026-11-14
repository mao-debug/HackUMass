"""CPU detector integration across local inputs, with no hardware or downloads."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vidi.ai import cpu_detection
from vidi.ai.object_detection import DetectionError
from vidi.core import controller
from vidi.core.configuration import (
    AppSettings, CameraSettings, ConfigurationError, ModelSettings, SimulationSettings,
    load_settings, validate_runtime_settings,
)
from vidi.core.types import Detection, DetectionResult, FramePacket
from vidi.main import main

ROOT = Path(__file__).resolve().parents[1]


def test_pi_profiles_select_distinct_camera_and_inference_contracts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ordinary = load_settings(ROOT / "config/pi.yaml")
    validate_runtime_settings(ordinary)
    assert ordinary.camera.source == "picamera2"
    assert ordinary.models.object_detection_backend == "opencv_ssd"
    assert ordinary.models.object_detection_path == ROOT / "models/mobilenet-ssd/mobilenet_iter_73000.caffemodel"
    assert ordinary.models.object_detection_config_path == ROOT / "models/mobilenet-ssd/deploy.prototxt"
    sensor = load_settings(ROOT / "config/pi-imx500.yaml")
    validate_runtime_settings(sensor)
    assert sensor.camera.source == "imx500"
    assert sensor.models.object_detection_backend == "imx500"


@pytest.mark.parametrize("arguments", [
    ["--detector", "opencv_ssd"],
    ["--detector", "opencv_ssd", "--model", "weights.caffemodel"],
    ["--detector", "opencv_ssd", "--model-config", "deploy.prototxt"],
    ["--source", "imx500", "--detector", "opencv_ssd", "--model", "weights.caffemodel", "--model-config", "deploy.prototxt"],
])
def test_incomplete_or_incompatible_cpu_selection_fails_before_capture(monkeypatch, arguments):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main(arguments) == 1


def test_missing_local_model_fails_before_camera_start(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main([
        "--source", "picamera2", "--detector", "opencv_ssd",
        "--model", str(tmp_path / "absent.caffemodel"),
        "--model-config", str(tmp_path / "absent.prototxt"),
    ]) == 1


def test_picamera_cpu_run_uses_each_owned_frame_and_reports_empty_last_result(monkeypatch, tmp_path):
    frame1 = np.full((48, 64, 3), (10, 20, 30), dtype=np.uint8)
    frame2 = np.full((48, 64, 3), (40, 50, 60), dtype=np.uint8)
    packets = [FramePacket(1, 1_000_000_000, frame1), FramePacket(2, 2_000_000_000, frame2)]
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.side_effect = packets
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    received = []

    class Detector:
        def __init__(self, models, settings):
            assert models.object_detection_backend == "opencv_ssd"

        def detect(self, packet):
            received.append(packet)
            objects = (Detection(15, "person", 0.9, (2, 3, 30, 40)),) if packet.frame_id == 1 else ()
            return DetectionResult(packet.frame_id, "opencv_ssd", "ready", objects)

    monkeypatch.setattr(cpu_detection, "OpenCvSsdDetector", Detector)
    settings = load_settings(ROOT / "config/pi.yaml")
    report = tmp_path / "cpu.json"
    settings = replace(settings, simulation=SimulationSettings(max_frames=2, report_path=report))
    result = controller.run_simulation(settings)
    assert received == packets
    assert (result.ready_frames, result.not_ready_frames, result.simulated) == (2, 0, False)
    assert result.detections == ()
    manager.__exit__.assert_called_once()
    np.testing.assert_array_equal(frame1[0, 0], (10, 20, 30))
    data = json.loads(report.read_text())
    assert data["source"] == "picamera2" and data["backend"] == "opencv_ssd"
    assert data["frames"][0]["inference"]["detections"][0]["label"] == "person"
    assert data["frames"][1]["inference"]["detections"] == []


def test_camera_only_picamera_run_does_not_construct_detector(monkeypatch):
    monkeypatch.setattr(cpu_detection, "OpenCvSsdDetector", lambda *args: pytest.fail("Detector constructed"))
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.return_value = FramePacket(
        1, 10, np.zeros((48, 64, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    result = controller.run_simulation(AppSettings(camera=CameraSettings(source="picamera2")))
    assert result.backend == "none" and result.ready_frames == 0
    assert result.detections == ()


def test_cpu_inference_failure_closes_camera(monkeypatch):
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.return_value = FramePacket(
        1, 10, np.zeros((48, 64, 3), dtype=np.uint8)
    )
    detector = MagicMock()
    detector.detect.side_effect = DetectionError("Model failed")
    monkeypatch.setattr(cpu_detection, "OpenCvSsdDetector", lambda *args: detector)
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    settings = replace(load_settings(ROOT / "config/pi.yaml"), simulation=SimulationSettings())
    with pytest.raises(DetectionError, match="Model failed"):
        controller.run_simulation(settings)
    manager.__exit__.assert_called_once()


def test_cli_resolves_cpu_model_config_path_from_working_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(controller, "run_simulation", lambda settings: calls.append(settings) or MagicMock(ready_frames=1))
    assert main([
        "--source", "sample", "--detector", "opencv_ssd",
        "--model", "weights.caffemodel", "--model-config", "deploy.prototxt",
    ]) == 0
    assert calls[0].models.object_detection_config_path == tmp_path / "deploy.prototxt"


def test_fixture_cannot_be_selected_for_picamera():
    settings = AppSettings(
        camera=CameraSettings(source="picamera2"),
        models=ModelSettings(object_detection_backend="fixture"),
    )
    with pytest.raises(ConfigurationError, match="never a live camera"):
        validate_runtime_settings(settings)
