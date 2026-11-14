"""Frame-source checks that never require a real camera or display."""

from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.camera.capture import CaptureError, FrameCapture
from vidi.core.configuration import CameraSettings


def test_sample_is_deterministic_and_returns_independent_frames() -> None:
    settings = CameraSettings(source="sample", width=160, height=120)
    with FrameCapture(settings) as capture:
        first = capture.read()
        second = capture.read()
        assert first is not None
        assert second is not None
        assert first.shape == (120, 160, 3)
        assert first.dtype == np.uint8
        np.testing.assert_array_equal(first, second)
        first[:] = 0
        assert np.any(second)
    with FrameCapture(settings) as capture:
        np.testing.assert_array_equal(capture.read(), second)


def test_image_source_decodes_and_repeats_without_mutating_original(tmp_path) -> None:
    path = tmp_path / "sample.png"
    original = np.full((12, 20, 3), (20, 80, 160), dtype=np.uint8)
    assert cv2.imwrite(str(path), original)
    with FrameCapture(CameraSettings(source="image", image_path=path)) as capture:
        first = capture.read()
        assert first is not None
        np.testing.assert_array_equal(first, original)
        first[:] = 0
        np.testing.assert_array_equal(capture.read(), original)


def test_missing_image_has_actionable_error(tmp_path) -> None:
    with pytest.raises(CaptureError, match="Image file does not exist"):
        with FrameCapture(
            CameraSettings(source="image", image_path=tmp_path / "missing.png")
        ):
            pass


def test_invalid_image_has_decode_error(tmp_path) -> None:
    path = tmp_path / "invalid.png"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(CaptureError, match="Could not decode image"):
        with FrameCapture(CameraSettings(source="image", image_path=path)):
            pass


def test_unavailable_webcam_is_released(monkeypatch) -> None:
    camera = Mock()
    camera.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=camera))
    with pytest.raises(CaptureError, match="Could not open webcam 2"):
        with FrameCapture(CameraSettings(source="webcam", webcam_index=2)):
            pass
    camera.release.assert_called_once_with()


def test_webcam_configuration_and_release_on_caller_error(monkeypatch) -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.return_value = (True, frame)
    factory = Mock(return_value=camera)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    settings = CameraSettings(
        source="webcam", webcam_index=3, width=320, height=240, fps=10.0
    )
    with pytest.raises(RuntimeError, match="caller failed"):
        with FrameCapture(settings) as capture:
            assert capture.read() is frame
            raise RuntimeError("caller failed")
    factory.assert_called_once_with(3)
    camera.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    camera.set.assert_any_call(cv2.CAP_PROP_FPS, 10.0)
    camera.release.assert_called_once_with()


def test_webcam_read_failure_returns_none(monkeypatch) -> None:
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.return_value = (False, None)
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=camera))
    with FrameCapture(CameraSettings(source="webcam")) as capture:
        assert capture.read() is None
    camera.release.assert_called_once_with()


def test_opencv_setup_error_releases_webcam(monkeypatch) -> None:
    camera = Mock()
    camera.isOpened.return_value = True
    camera.set.side_effect = cv2.error("camera setup failed")
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=camera))
    with pytest.raises(CaptureError, match="OpenCV could not open"):
        with FrameCapture(CameraSettings(source="webcam")):
            pass
    camera.release.assert_called_once_with()


def test_opencv_read_error_becomes_capture_error_and_releases_camera(monkeypatch) -> None:
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.side_effect = cv2.error("camera read failed")
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=camera))
    capture = FrameCapture(CameraSettings(source="webcam"))
    with pytest.raises(CaptureError, match="OpenCV could not read"):
        capture.read()
    capture.close()
    camera.release.assert_called_once_with()


def test_close_is_idempotent_and_read_after_close_fails() -> None:
    capture = FrameCapture(CameraSettings())
    capture.read()
    capture.close()
    capture.close()
    with pytest.raises(CaptureError, match="frame source is closed"):
        capture.read()
