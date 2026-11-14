"""Camera-only Picamera2 adapter for ordinary Raspberry Pi camera modules.

No IMX500 helper, model firmware, or inference engine is imported or selected.
The public adapter shares bounded process transport with the AI Camera adapter.
"""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from vidi.camera.capture import CaptureError
from vidi.camera.managed import ManagedPiCapture, run_worker
from vidi.core.types import FramePacket

if TYPE_CHECKING:
    from vidi.core.configuration import CameraSettings

logger = logging.getLogger(__name__)


def _load_picamera2():
    """Import only the ordinary camera API, exclusively in a hardware worker."""
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise CaptureError(
            "Pi camera capture needs the Raspberry Pi OS Picamera2/libcamera packages. "
            "Use --source sample on a development computer."
        ) from exc
    return Picamera2


class _Picamera2Session:
    """Worker-local ordinary camera owner; camera index 0 is explicit."""

    def __init__(self, settings: CameraSettings) -> None:
        self.camera = None
        Picamera2 = _load_picamera2()
        try:
            self.camera = Picamera2(0)
            configuration = self.camera.create_preview_configuration(
                main={"size": (settings.width, settings.height), "format": "RGB888"},
                controls={"FrameRate": settings.fps}, buffer_count=4, queue=False,
            )
            # Picamera2's RGB888 arrays contain B,G,R bytes, matching OpenCV.
            self.camera.configure(configuration)
            self.camera.start(show_preview=False)
            self._prepare_snapshot(settings)
        except BaseException:
            self.close()
            raise

    def _prepare_snapshot(self, settings: CameraSettings) -> None:
        """Apply opt-in settling/focus while the parent startup deadline runs.

        Picamera2 0.3.27 supports asynchronous autofocus_cycle and wait(timeout).
        A timed-out job must be cancelled and the camera closed rather than
        placing a capture request behind an unfinished autofocus operation.
        """
        if settings.settle_s:
            time.sleep(settings.settle_s)
        if settings.autofocus == "off":
            return
        try:
            controls = self.camera.camera_controls
            if not {"AfMode", "AfTrigger"}.issubset(controls):
                logger.warning("Pi camera autofocus controls are unavailable; attempting a snapshot without autofocus.")
                return
            job = self.camera.autofocus_cycle(wait=False)
            focused = self.camera.wait(job, timeout=settings.autofocus_timeout_s)
        except FutureTimeoutError as exc:
            self._cancel_autofocus()
            raise CaptureError(
                f"Pi camera autofocus timed out after {settings.autofocus_timeout_s:g} seconds; "
                "no snapshot was captured."
            ) from exc
        except Exception as exc:
            self._cancel_autofocus()
            raise CaptureError("Pi camera autofocus failed; no snapshot was captured.") from exc
        if focused:
            logger.info("Pi camera reported autofocus success; text sharpness has not been verified.")
        else:
            logger.warning("Pi camera autofocus did not confirm focus; attempting a snapshot without confirmed focus.")

    def _cancel_autofocus(self) -> None:
        """Best-effort queue cleanup; preserve the original autofocus failure."""
        try:
            self.camera.cancel_all_and_flush()
        except Exception as exc:
            logger.warning("Pi camera could not cancel pending autofocus operations: %s", exc)

    def read_packet(self, frame_id: int) -> FramePacket | None:
        request = self.camera.capture_request(flush=True)
        if request is None:
            return None
        try:
            image = np.array(request.make_array("main"), copy=True, order="C")
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
                raise CaptureError("Pi camera did not provide the configured nonempty uint8 BGR image.")
            return FramePacket(frame_id, time.monotonic_ns(), image, inference=None)
        finally:
            request.release()

    def close(self) -> None:
        camera, self.camera = self.camera, None
        if camera is not None:
            for action in (camera.stop, camera.close):
                try:
                    action()
                except Exception as exc:
                    logger.warning("Pi camera cleanup could not complete a camera operation: %s", exc)


def _worker_main(connection, shared_name: str, frame_bytes: int, settings) -> None:
    """Construct the ordinary session inside its dedicated spawned process."""
    run_worker(connection, shared_name, frame_bytes, _Picamera2Session, (settings,))


class Picamera2Capture(ManagedPiCapture):
    """Read owned camera frames with finite waits and no local inference."""

    def __init__(self, camera_settings: CameraSettings) -> None:
        super().__init__(
            camera_settings, _worker_main, (camera_settings,), source_name="Picamera2",
            startup_message="Starting ordinary Pi camera capture at camera index 0; inference runs separately.",
            ready_message="Picamera2 worker ready for camera-only capture; inference runs separately.",
        )
