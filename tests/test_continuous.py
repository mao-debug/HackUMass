"""Continuous local capture stays bounded in memory and has explicit stop paths."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

from vidi.core import controller
from vidi.core.configuration import AppSettings, CameraSettings, DetectionSettings, ModelSettings, SimulationSettings
from vidi.core.types import Detection, DetectionResult


def install_source(monkeypatch, frames):
    source = Mock()
    source.read.side_effect = frames
    manager = MagicMock()
    manager.__enter__.return_value = source

    def close_source(*args):
        source.close()
        return False

    manager.__exit__.side_effect = close_source
    factory = Mock(return_value=manager)
    monkeypatch.setattr(controller, "_open_source", factory)
    monkeypatch.setattr(controller.time, "sleep", Mock())
    return source, manager, factory


def preview_stubs(monkeypatch, keys=None):
    monkeypatch.setenv("DISPLAY", ":test")
    show, close = Mock(), Mock()
    key = Mock(side_effect=keys) if keys is not None else Mock(return_value=-1)
    monkeypatch.setattr(controller.cv2, "imshow", show)
    monkeypatch.setattr(controller.cv2, "waitKey", key)
    monkeypatch.setattr(controller.cv2, "destroyAllWindows", close)
    return show, close


def fixture_settings(tmp_path, *, preview=False, export=True):
    fixture = tmp_path / "detections.json"
    fixture.write_text(json.dumps({
        "version": 1, "image_size": [80, 60],
        "detections": [{"class_id": 9, "label": "chair", "confidence": 0.8, "bbox_xyxy": [10, 12, 40, 45]}],
    }))
    return AppSettings(
        camera=CameraSettings(width=80, height=60),
        models=ModelSettings(object_detection_backend="fixture"),
        detection=DetectionSettings(fixture_path=fixture),
        simulation=SimulationSettings(
            max_frames=1, preview=preview,
            output_path=tmp_path / "last.png" if export else None,
            report_path=tmp_path / "run.json",
        ),
    )


@pytest.mark.parametrize("exit_key", [ord("q"), 27])
def test_preview_stop_ignores_frame_limit_detects_each_frame_and_opens_camera_once(monkeypatch, tmp_path, exit_key):
    frames = [np.full((60, 80, 3), value, np.uint8) for value in (10, 20, 30)]
    source, manager, factory = install_source(monkeypatch, frames)
    show, close = preview_stubs(monkeypatch, [-1, -1, exit_key])
    settings = fixture_settings(tmp_path, preview=True)
    original_save = controller._save_image

    def save_after_cleanup(path, image):
        source.close.assert_called_once_with()
        close.assert_called_once_with()
        original_save(path, image)

    monkeypatch.setattr(controller, "_save_image", save_after_cleanup)

    result = controller.run_simulation(settings, continuous=True)

    assert result.frames_processed == result.ready_frames == 3
    assert result.interrupted is False and result.simulated is True
    assert result.last_inference.frame_id == 3 and result.detections[0].label == "chair"
    factory.assert_called_once_with(settings)
    manager.__enter__.assert_called_once_with()
    manager.__exit__.assert_called_once()
    assert source.read.call_count == show.call_count == 3
    assert not np.array_equal(show.call_args.args[1], frames[-1])
    assert cv2.imread(str(result.output_path)) is not None
    report = json.loads(settings.simulation.report_path.read_text())
    assert report["frames_processed"] == 3 and report["continuous"] is True
    assert report["frame_records_limit"] == 120 and report["frame_records_truncated"] is False
    assert report["frame_records_dropped"] == 0 and report["interrupted"] is False
    assert [record["inference"]["frame_id"] for record in report["frames"]] == [1, 2, 3]


@pytest.mark.parametrize("preview", [False, True])
def test_ctrl_c_cleans_up_then_preserves_last_completed_frame_and_report(monkeypatch, tmp_path, caplog, preview):
    frames = [np.full((60, 80, 3), value, np.uint8) for value in (10, 20)]
    source, manager, factory = install_source(monkeypatch, [*frames, KeyboardInterrupt])
    _, close = preview_stubs(monkeypatch)
    settings = AppSettings(simulation=SimulationSettings(
        max_frames=1, preview=preview, output_path=tmp_path / "last.png", report_path=tmp_path / "run.json",
    ))
    caplog.set_level("INFO")

    result = controller.run_simulation(settings, continuous=True)

    assert result.interrupted is True and result.frames_processed == 2
    assert "Continuous local mode active" in caplog.text
    factory.assert_called_once_with(settings)
    manager.__exit__.assert_called_once()
    source.close.assert_called_once_with()
    assert close.call_count == int(preview)
    np.testing.assert_array_equal(cv2.imread(str(result.output_path)), frames[-1])
    report = json.loads(result.report_path.read_text())
    assert report["interrupted"] is True and report["frames_processed"] == 2
    assert [record["frame_id"] for record in report["frames"]] == [1, 2]


def test_continuous_report_history_is_bounded_while_totals_cover_entire_run(monkeypatch, tmp_path):
    frame = np.zeros((60, 80, 3), np.uint8)
    source, _, _ = install_source(monkeypatch, [frame] * 125 + [KeyboardInterrupt])
    settings = fixture_settings(tmp_path, export=False)

    result = controller.run_simulation(settings, continuous=True)

    assert result.interrupted is True
    assert result.frames_processed == result.ready_frames == 125
    assert result.not_ready_frames == 0 and result.last_inference.frame_id == 125
    assert source.read.call_count == 126
    report = json.loads(result.report_path.read_text())
    assert len(report["frames"]) == report["frame_records_limit"] == 120
    assert report["frame_records_truncated"] is True and report["frame_records_dropped"] == 5
    assert report["frames_processed"] == report["ready_frames"] == 125
    assert report["frames"][0]["frame_id"] == 6 and report["frames"][-1]["frame_id"] == 125


def test_interrupt_during_inference_does_not_count_or_export_unfinished_frame(monkeypatch, tmp_path):
    from vidi.ai import cpu_detection

    frames = [np.full((60, 80, 3), value, np.uint8) for value in (10, 20)]
    source, _, _ = install_source(monkeypatch, frames)
    detector = Mock()
    detector.detect.side_effect = [
        DetectionResult(1, "opencv_ssd", "ready", (Detection(9, "chair", 0.8, (10, 12, 40, 45)),)),
        KeyboardInterrupt,
    ]
    monkeypatch.setattr(cpu_detection, "OpenCvSsdDetector", lambda *args: detector)
    settings = AppSettings(
        models=ModelSettings(
            object_detection_backend="opencv_ssd", object_detection_path=Path("model.caffemodel"),
            object_detection_config_path=Path("model.prototxt"),
        ),
        simulation=SimulationSettings(output_path=tmp_path / "last.png", report_path=tmp_path / "run.json"),
    )

    result = controller.run_simulation(settings, continuous=True)

    assert result.interrupted is True and result.frames_processed == result.ready_frames == 1
    assert source.read.call_count == 2 and result.last_inference.frame_id == 1
    report = json.loads(result.report_path.read_text())
    assert len(report["frames"]) == 1 and report["frames"][0]["frame_id"] == 1
    np.testing.assert_array_equal(cv2.imread(str(result.output_path))[-1, -1], frames[0][-1, -1])


def test_initial_ctrl_c_propagates_without_inventing_a_final_frame(monkeypatch, tmp_path):
    source, manager, _ = install_source(monkeypatch, [KeyboardInterrupt])
    _, close = preview_stubs(monkeypatch)
    settings = AppSettings(simulation=SimulationSettings(
        preview=True, output_path=tmp_path / "last.png", report_path=tmp_path / "run.json",
    ))

    with pytest.raises(KeyboardInterrupt):
        controller.run_simulation(settings, continuous=True)

    manager.__exit__.assert_called_once()
    source.close.assert_called_once_with()
    close.assert_called_once_with()
    assert not settings.simulation.output_path.exists() and not settings.simulation.report_path.exists()


def test_bounded_mode_keeps_frame_limit_and_omits_continuous_history_metadata(monkeypatch, tmp_path):
    frame = np.zeros((60, 80, 3), np.uint8)
    source, _, _ = install_source(monkeypatch, [frame] * 3)
    settings = AppSettings(simulation=SimulationSettings(max_frames=2, report_path=tmp_path / "run.json"))

    result = controller.run_simulation(settings)

    assert source.read.call_count == result.frames_processed == 2 and result.interrupted is False
    report = json.loads(result.report_path.read_text())
    assert "continuous" not in report and "frame_records_limit" not in report
    assert "frame_records_truncated" not in report and "frame_records_dropped" not in report


def test_bounded_mode_still_propagates_ctrl_c_after_completed_frames(monkeypatch):
    source, manager, _ = install_source(monkeypatch, [np.zeros((60, 80, 3), np.uint8), KeyboardInterrupt])

    with pytest.raises(KeyboardInterrupt):
        controller.run_simulation(AppSettings(simulation=SimulationSettings(max_frames=3)))

    assert source.read.call_count == 2
    manager.__exit__.assert_called_once()
    source.close.assert_called_once_with()


def test_interrupted_export_failures_do_not_mask_stop_or_claim_artifacts_saved(monkeypatch, tmp_path, caplog):
    install_source(monkeypatch, [np.zeros((60, 80, 3), np.uint8), KeyboardInterrupt])
    monkeypatch.setattr(controller, "_save_image", Mock(side_effect=controller.SimulationError("Image export failed")))
    monkeypatch.setattr(controller, "_save_report", Mock(side_effect=controller.SimulationError("Report export failed")))
    settings = AppSettings(simulation=SimulationSettings(
        output_path=tmp_path / "last.png", report_path=tmp_path / "run.json",
    ))
    caplog.set_level("INFO")

    result = controller.run_simulation(settings, continuous=True)

    assert result.interrupted is True and result.output_path is None
    assert "Image export failed" in caplog.text and "Report export failed" in caplog.text
    assert "Last frame saved" not in caplog.text and "Run report saved" not in caplog.text
