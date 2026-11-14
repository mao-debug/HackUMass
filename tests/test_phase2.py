"""Phase 2 integration contracts, with all hardware and cloud calls replaced."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from vidi.camera.capture import FrameCapture
from vidi.core import controller
from vidi.core.configuration import (
    AppSettings, CameraSettings, ConfigurationError, DetectionSettings,
    ModelSettings, SimulationSettings, load_settings, validate_runtime_settings,
)
from vidi.core.types import Detection, DetectionResult, FramePacket
from vidi.main import main

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_still_matches_hardware_free_defaults():
    assert load_settings(ROOT / "config/settings.yaml") == AppSettings()


def test_fixture_run_exports_annotated_image_and_truthful_report(tmp_path):
    output, report = tmp_path / "frame.png", tmp_path / "run.json"
    assert main([
        "--config", str(ROOT / "config/fixtures.yaml"),
        "--output", str(output), "--report", str(report),
    ]) == 0
    data = json.loads(report.read_text())
    assert data["backend"] == "fixture" and data["simulated"] is True
    assert data["ready_frames"] == 1 and data["description_status"] == "not_requested"
    assert {d["label"] for d in data["detections"]} == {"person", "chair"}
    assert data["frames"][0]["inference"]["frame_id"] == data["frames"][0]["frame_id"]
    assert data["startup_s"] >= 0 and data["capture_fps"] is None
    rendered = cv2.imread(str(output))
    with FrameCapture(CameraSettings()) as source:
        raw = source.read()
    assert rendered.shape == raw.shape
    assert not np.array_equal(rendered, raw)


def test_configured_gemini_and_key_do_not_enable_cloud(monkeypatch, tmp_path):
    from vidi.ai import scene_description

    def forbidden(*args, **kwargs):
        raise AssertionError("A normal run must not construct a cloud client")

    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", forbidden)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    settings = AppSettings(models=ModelSettings(scene_description_backend="gemini"))
    result = controller.run_simulation(settings)
    assert result.description_status == "not_requested"


def test_explicit_description_uses_unannotated_frame_after_capture_closed(monkeypatch, tmp_path):
    from vidi.ai import scene_description

    settings = load_settings(ROOT / "config/fixtures.yaml")
    settings = replace(settings, simulation=SimulationSettings(
        output_path=tmp_path / "annotated.png", report_path=tmp_path / "report.json"
    ))
    with FrameCapture(settings.camera) as source:
        raw = source.read()
    manager = MagicMock()
    manager.__enter__.return_value.read.return_value = raw
    monkeypatch.setattr(controller, "FrameCapture", lambda camera: manager)

    class FakeDescriber:
        def __init__(self, cloud):
            pass

        def describe(self, image, question):
            manager.__exit__.assert_called_once()
            np.testing.assert_array_equal(image, raw)
            assert question == "Describe the scene."
            return scene_description.SceneDescriptionResult("A controlled test scene.", "test-model", 0.01)

    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", FakeDescriber)
    result = controller.run_simulation(settings, describe=True, question="Describe the scene.")
    assert result.description_status == "ready"
    assert result.description_model == "test-model"
    assert len(result.detections) == 2


def test_cloud_failure_preserves_artifacts_and_returns_nonzero(monkeypatch, tmp_path):
    from vidi.ai import scene_description

    class Unavailable:
        def __init__(self, cloud):
            pass

        def describe(self, image, question):
            raise scene_description.SceneDescriptionError("Network unavailable")

    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", Unavailable)
    output, report = tmp_path / "frame.png", tmp_path / "run.json"
    assert main([
        "--config", str(ROOT / "config/fixtures.yaml"), "--describe",
        "--output", str(output), "--report", str(report),
    ]) == 1
    data = json.loads(report.read_text())
    assert output.is_file()
    assert data["description_status"] == "unavailable"
    assert data["description_error"] == "Network unavailable"
    assert len(data["detections"]) == 2


def test_empty_and_unavailable_imx500_results_never_reuse_boxes(monkeypatch, tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    found = Detection(0, "person", 0.9, (1, 2, 30, 40))
    packets = [
        FramePacket(1, 10, frame, DetectionResult(1, "imx500", "ready", (found,))),
        FramePacket(2, 20, frame, DetectionResult(2, "imx500", "ready")),
        FramePacket(3, 30, frame, DetectionResult(3, "imx500", "not_ready")),
    ]
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.side_effect = packets
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    report = tmp_path / "imx500.json"
    settings = AppSettings(
        camera=CameraSettings(source="imx500"),
        models=ModelSettings(object_detection_backend="imx500", object_detection_path=Path("model.rpk")),
        simulation=SimulationSettings(max_frames=3, report_path=report),
    )
    result = controller.run_simulation(settings)
    assert result.ready_frames == 2 and result.not_ready_frames == 1
    assert result.detections == ()
    assert result.simulated is False
    manager.__exit__.assert_called_once()
    data = json.loads(report.read_text())
    assert data["frames"][1]["inference"]["detections"] == []
    assert data["last_inference_status"] == "not_ready"


def test_mismatched_frame_metadata_is_rejected_and_source_closed(monkeypatch):
    from vidi.ai.object_detection import DetectionError

    manager = MagicMock()
    manager.__enter__.return_value.read_packet.return_value = FramePacket(
        2, 20, np.zeros((48, 64, 3), dtype=np.uint8), DetectionResult(1, "imx500", "ready")
    )
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    settings = AppSettings(
        camera=CameraSettings(source="imx500"),
        models=ModelSettings(object_detection_backend="imx500", object_detection_path=Path("model.rpk")),
    )
    with pytest.raises(DetectionError, match="different frame"):
        controller.run_simulation(settings)
    manager.__exit__.assert_called_once()


@pytest.mark.parametrize("arguments", [
    ["--detector", "unknown"], ["--detector", "imx500"],
    ["--detector", "fixture"], ["--question", "Describe this"],
    ["--threshold", "nan"], ["--threshold", "inf"],
    ["--threshold", "-0.1"], ["--threshold", "1.1"], ["--max-detections", "0"],
])
def test_invalid_phase2_options_never_open_camera(monkeypatch, arguments):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    if "unknown" in arguments:
        with pytest.raises(SystemExit) as exc:
            main(arguments)
        assert exc.value.code == 2
    else:
        assert main(arguments) == 1


@pytest.mark.parametrize("content", [
    "camera: {startup_timeout_s: 0}", "camera: {capture_timeout_s: .nan}",
    "camera: {capture_timeout_s: true}", "detection: {confidence_threshold: true}",
    "detection: {confidence_threshold: .nan}", "detection: {max_detections: false}",
    "detection: {labels_path: []}", "cloud: {model: null}", "cloud: {model: []}",
    "cloud: {timeout_s: .inf}", "cloud: {max_output_tokens: 0}",
    "cloud: {max_image_edge: 1025}", "cloud: {max_image_edge: true}",
    "simulation: {report_path: false}", "detection: {unknown: true}",
])
def test_invalid_phase2_yaml_fails_cleanly(tmp_path, content):
    path = tmp_path / "settings.yaml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        load_settings(path)


def test_json_report_cannot_overwrite_image_output(tmp_path):
    settings = AppSettings(simulation=SimulationSettings(
        output_path=tmp_path / "frame.png", report_path=tmp_path / "frame.png"
    ))
    with pytest.raises(ConfigurationError, match="different paths"):
        validate_runtime_settings(settings)


def test_fixture_is_forbidden_on_live_sources():
    settings = AppSettings(
        camera=CameraSettings(source="webcam"),
        models=ModelSettings(object_detection_backend="fixture"),
        detection=DetectionSettings(fixture_path=ROOT / "examples/detection-fixture.json"),
    )
    with pytest.raises(ConfigurationError, match="never a live camera"):
        validate_runtime_settings(settings)


def test_all_not_ready_run_is_not_reported_as_success(monkeypatch, tmp_path, caplog):
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.return_value = FramePacket(
        1, 10, np.zeros((48, 64, 3), dtype=np.uint8), DetectionResult(1, "imx500", "not_ready")
    )
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    report = tmp_path / "warming.json"
    assert main([
        "--config", str(ROOT / "config/pi-imx500.yaml"), "--max-frames", "1", "--report", str(report)
    ]) == 1
    assert "No inference results arrived" in caplog.text
    assert json.loads(report.read_text())["last_inference_status"] == "not_ready"
    manager.__exit__.assert_called_once()


def test_processing_rate_excludes_camera_shutdown(monkeypatch):
    clock = {"now": 0.0}
    source = MagicMock()

    def enter():
        clock["now"] = 1.0
        return source

    def read():
        clock["now"] = 2.0
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def shutdown(*args):
        clock["now"] = 100.0

    source.__enter__.side_effect = enter
    source.read.side_effect = read
    source.__exit__.side_effect = shutdown
    monkeypatch.setattr(controller, "_open_source", lambda settings: source)
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock["now"])
    result = controller.run_simulation(AppSettings())
    assert result.startup_s == 1.0
    assert result.processing_s == 1.0


def test_report_write_failure_is_controlled(tmp_path, caplog):
    parent = tmp_path / "file"
    parent.write_text("not a directory")
    assert main(["--report", str(parent / "report.json")]) == 1
    assert "Cannot save JSON report" in caplog.text
    assert "Traceback" not in caplog.text


def test_oversized_question_is_rejected_before_capture(monkeypatch):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main(["--describe", "--question", "x" * 501]) == 1


def test_observed_rates_use_frame_intervals_and_require_two_observations(monkeypatch, tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    packets = [
        FramePacket(1, 1_000_000_000, frame, DetectionResult(1, "imx500", "ready")),
        FramePacket(2, 1_200_000_000, frame, DetectionResult(2, "imx500", "not_ready")),
        FramePacket(3, 1_400_000_000, frame, DetectionResult(3, "imx500", "ready")),
    ]
    manager = MagicMock()
    manager.__enter__.return_value.read_packet.side_effect = packets
    monkeypatch.setattr(controller, "_open_source", lambda settings: manager)
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    report = tmp_path / "rates.json"
    settings = AppSettings(
        camera=CameraSettings(source="imx500"),
        models=ModelSettings(object_detection_backend="imx500", object_detection_path=Path("model.rpk")),
        simulation=SimulationSettings(max_frames=3, report_path=report),
    )
    controller.run_simulation(settings)
    data = json.loads(report.read_text())
    assert data["capture_fps"] == pytest.approx(5.0)
    assert data["inference_ready_fps"] == pytest.approx(2.5)
