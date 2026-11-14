"""Portable frame sources; the AI Camera adapter lives in camera.imx500.

Frames use OpenCV's BGR color order. The built-in sample is a synthetic test
pattern, not a photograph or the output of an object detection model.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from vidi.core.configuration import CameraSettings

logger = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """A requested frame source could not be opened or read."""


class FrameCapture:
    """Read repeatable sample/image frames or frames from an optional webcam.

    Opening is lazy: use a context manager or call ``read`` to open the source.
    Always call ``close`` when finished, preferably through the context manager.
    A webcam that stops delivering frames returns ``None`` from ``read``.
    """

    def __init__(self, settings: CameraSettings) -> None:
        self.settings = settings
        self._camera: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._opened = False
        self._closed = False

    def __enter__(self) -> FrameCapture:
        self._open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _open(self) -> None:
        if self._closed:
            raise CaptureError("This frame source is closed. Create a new FrameCapture.")
        if self._opened:
            return

        try:
            if self.settings.source == "sample":
                self._frame = self._make_sample()
            elif self.settings.source == "image":
                image_path = self.settings.image_path
                if image_path is None or not image_path.is_file():
                    raise CaptureError(f"Image file does not exist: {image_path}")
                self._frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if self._frame is None or self._frame.size == 0:
                    raise CaptureError(f"Could not decode image file: {image_path}")
            elif self.settings.source == "webcam":
                self._camera = cv2.VideoCapture(self.settings.webcam_index)
                if not self._camera.isOpened():
                    raise CaptureError(
                        f"Could not open webcam {self.settings.webcam_index}. "
                        "Check the connection and camera permission, or use --source sample."
                    )
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
                self._camera.set(cv2.CAP_PROP_FPS, self.settings.fps)
            else:
                raise CaptureError(f"Unsupported frame source: {self.settings.source}")
        except cv2.error as exc:
            self.close()
            raise CaptureError(f"OpenCV could not open the frame source: {exc}") from exc
        except CaptureError:
            self.close()
            raise

        self._opened = True

    def read(self) -> np.ndarray | None:
        """Return a BGR frame, or ``None`` if the webcam has no further frame."""
        self._open()
        if self._frame is not None:
            # Do not allow preview annotations or downstream edits to change the source.
            return self._frame.copy()
        assert self._camera is not None
        try:
            success, frame = self._camera.read()
        except cv2.error as exc:
            self.close()
            raise CaptureError(f"OpenCV could not read the webcam: {exc}") from exc
        if not success or frame is None or frame.size == 0:
            return None
        return frame

    def close(self) -> None:
        """Release the webcam; safe to call more than once, including after errors."""
        camera, self._camera = self._camera, None
        self._frame = None
        self._opened = False
        self._closed = True
        if camera is not None:
            try:
                camera.release()
            except cv2.error as exc:
                logger.warning("OpenCV could not release the webcam cleanly: %s", exc)

    def _make_sample(self) -> np.ndarray:
        """Build a deterministic labeled test pattern without hardware or assets."""
        width, height = self.settings.width, self.settings.height
        frame = np.full((height, width, 3), (38, 35, 30), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (width // 10, height // 3),
            (width * 4 // 10, height * 4 // 5),
            (185, 120, 55),
            thickness=-1,
        )
        cv2.circle(
            frame,
            (width * 7 // 10, height * 3 // 5),
            max(1, min(width, height) // 7),
            (65, 170, 220),
            thickness=-1,
        )
        font_scale = max(0.2, min(width / 1000, height / 600))
        cv2.putText(
            frame,
            "SYNTHETIC SAMPLE - NO INFERENCE",
            (max(2, width // 30), max(10, height // 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        return frame
