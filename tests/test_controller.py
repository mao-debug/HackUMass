"""Simulation orchestration checks without physical cameras or GUI windows."""

from unittest.mock import MagicMock, Mock, call

import cv2
import numpy as np
import pytest

from vidi.camera.capture import CaptureError
from vidi.core import controller
from vidi.core.configuration import AppSettings, CameraSettings, SimulationSettings
from vidi.core.controller import SimulationError, run_simulation


def install_capture(monkeypatch, frames):
    """Install a context-managed source and record its cleanup."""
    capture = Mock()
    capture.read.side_effect = frames
    manager = MagicMock()
    manager.__enter__.return_value = capture

    def exit_context(*args):
        capture.close()
        return False

    manager.__exit__.side_effect = exit_context
    factory = Mock(return_value=manager)
    monkeypatch.setattr(controller, "FrameCapture", factory)
    return capture, manager, factory


def allow_mock_preview(monkeypatch):
    """Permit preview on all hosts while replacing every GUI operation."""
    monkeypatch.setenv("DISPLAY", ":test")
    imshow = Mock()
    wait_key = Mock(return_value=-1)
    destroy = Mock()
    monkeypatch.setattr(controller.cv2, "imshow", imshow)
    monkeypatch.setattr(controller.cv2, "waitKey", wait_key)
    monkeypatch.setattr(controller.cv2, "destroyAllWindows", destroy)
    return imshow, wait_key, destroy


def test_default_sample_returns_observable_result() -> None:
    result = run_simulation(AppSettings())
    assert result.frames_processed == 1
    assert (result.width, result.height) == (640, 480)
    assert result.output_path is None


def test_output_exports_last_sample_frame_and_can_be_read_back(tmp_path) -> None:
    output = tmp_path / "new-directory" / "sample.png"
    settings = AppSettings(
        camera=CameraSettings(width=160, height=120),
        simulation=SimulationSettings(output_path=output),
    )
    result = run_simulation(settings)
    saved = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert result.output_path == output
    assert saved is not None
    assert saved.shape == (120, 160, 3)
    assert np.any(saved)


@pytest.mark.parametrize("successful_frames", [0, 1])
def test_source_stopping_raises_capture_error_and_closes(monkeypatch, successful_frames) -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    capture, manager, _ = install_capture(
        monkeypatch, [frame] * successful_frames + [None]
    )
    monkeypatch.setattr(controller.time, "sleep", Mock())
    settings = AppSettings(simulation=SimulationSettings(max_frames=2))
    with pytest.raises(CaptureError, match=f"after {successful_frames} frame"):
        run_simulation(settings)
    assert capture.read.call_count == successful_frames + 1
    manager.__exit__.assert_called_once()
    capture.close.assert_called_once_with()


def test_preview_error_closes_source_and_window(monkeypatch) -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    capture, manager, _ = install_capture(monkeypatch, [frame])
    imshow, _, destroy = allow_mock_preview(monkeypatch)
    imshow.side_effect = cv2.error("display unavailable")
    settings = AppSettings(simulation=SimulationSettings(preview=True))
    with pytest.raises(SimulationError, match="OpenCV preview failed"):
        run_simulation(settings)
    manager.__exit__.assert_called_once()
    capture.close.assert_called_once_with()
    destroy.assert_called_once_with()


def test_linux_without_display_rejects_preview_before_opening_camera(monkeypatch) -> None:
    monkeypatch.setattr(controller.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    factory = Mock()
    monkeypatch.setattr(controller, "FrameCapture", factory)
    settings = AppSettings(simulation=SimulationSettings(preview=True))
    with pytest.raises(SimulationError, match="Preview needs a desktop display"):
        run_simulation(settings)
    factory.assert_not_called()


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_preview_exit_key_stops_early_and_cleans_up(monkeypatch, key) -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    capture, manager, _ = install_capture(monkeypatch, [frame])
    imshow, wait_key, destroy = allow_mock_preview(monkeypatch)
    wait_key.return_value = key
    sleep = Mock()
    monkeypatch.setattr(controller.time, "sleep", sleep)
    settings = AppSettings(simulation=SimulationSettings(max_frames=5, preview=True))
    result = run_simulation(settings)
    assert result.frames_processed == 1
    assert (result.width, result.height) == (20, 12)
    capture.read.assert_called_once_with()
    manager.__exit__.assert_called_once()
    capture.close.assert_called_once_with()
    imshow.assert_called_once()
    destroy.assert_called_once_with()
    sleep.assert_not_called()


def test_multiple_frames_are_bounded_and_paced(monkeypatch) -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    capture, _, _ = install_capture(monkeypatch, [frame, frame, frame])
    sleep = Mock()
    monkeypatch.setattr(controller.time, "sleep", sleep)
    monkeypatch.setattr(controller.time, "monotonic", Mock(return_value=100.0))
    result = run_simulation(
        AppSettings(
            camera=CameraSettings(fps=10.0),
            simulation=SimulationSettings(max_frames=3),
        )
    )
    assert result.frames_processed == 3
    assert capture.read.call_count == 3
    assert sleep.call_args_list == [call(0.1), call(0.1)]
    capture.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["false", "opencv-error"])
def test_output_encoder_failures_raise_simulation_error(monkeypatch, tmp_path, failure) -> None:
    writer = Mock(return_value=False)
    if failure == "opencv-error":
        writer.side_effect = cv2.error("encoder unavailable")
    monkeypatch.setattr(controller.cv2, "imwrite", writer)
    settings = AppSettings(
        simulation=SimulationSettings(output_path=tmp_path / "sample.png")
    )
    with pytest.raises(SimulationError, match="[Cc]ould not save|Cannot save"):
        run_simulation(settings)
    writer.assert_called_once()


def test_output_directory_failure_raises_simulation_error(tmp_path) -> None:
    parent_file = tmp_path / "existing-file"
    parent_file.write_text("a file, not a directory", encoding="utf-8")
    settings = AppSettings(
        simulation=SimulationSettings(output_path=parent_file / "sample.png")
    )
    with pytest.raises(SimulationError, match="Cannot save frame"):
        run_simulation(settings)
