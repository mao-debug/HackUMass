"""Bounded local simulation; no inference, audio, or networking in Phase 1."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from visionbridge.camera.capture import CaptureError, FrameCapture
from visionbridge.core.configuration import AppSettings

logger = logging.getLogger(__name__)
WINDOW_TITLE = "VisionBridge - simulation only (q/Esc to exit)"


class SimulationError(RuntimeError):
    """The simulation could not display or export a frame."""


@dataclass(frozen=True)
class SimulationResult:
    """An observable result; frame dimensions reflect actual source output."""

    frames_processed: int
    width: int
    height: int
    output_path: Path | None


def run_simulation(settings: AppSettings) -> SimulationResult:
    """Read a limited number of frames and optionally preview/save the last.

    The configured FPS limits the processing loop; webcam drivers may provide
    different dimensions or capture rates. No scene understanding is attempted.
    """
    preview = settings.simulation.preview
    if preview and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise SimulationError("Preview needs a desktop display. Run with --no-preview instead.")

    count = 0
    last_frame = None
    try:
        with FrameCapture(settings.camera) as capture:
            for _ in range(settings.simulation.max_frames):
                started = time.monotonic()
                frame = capture.read()
                if frame is None:
                    raise CaptureError(
                        f"Frame source stopped delivering frames after {count} frame(s). "
                        "Check the webcam or run with --source sample."
                    )
                last_frame = frame
                count += 1
                logger.debug("Frame %d: %dx%d", count, frame.shape[1], frame.shape[0])
                if preview:
                    cv2.imshow(WINDOW_TITLE, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if count < settings.simulation.max_frames:
                    time.sleep(max(0.0, 1.0 / settings.camera.fps - (time.monotonic() - started)))
    except cv2.error as exc:
        raise SimulationError(f"OpenCV preview failed; try --no-preview. Details: {exc}") from exc
    finally:
        if preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                logger.debug("OpenCV had no preview windows to close.")

    assert last_frame is not None  # max_frames is validated to be positive.
    output = settings.simulation.output_path
    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), last_frame):
                raise SimulationError(f"OpenCV could not save the frame to {output}.")
        except (OSError, cv2.error) as exc:
            raise SimulationError(f"Cannot save frame to {output}: {exc}") from exc

    result = SimulationResult(count, last_frame.shape[1], last_frame.shape[0], output)
    logger.info(
        "Simulation completed: %d frame(s), %dx%d, source=%s. AI and audio are not active.",
        result.frames_processed, result.width, result.height, settings.camera.source,
    )
    if output:
        logger.info("Last frame saved to %s", output)
    return result
