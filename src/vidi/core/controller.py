"""Bounded local capture/detection, reporting, and an explicit cloud opt-in."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from vidi.camera.capture import CaptureError, FrameCapture
from vidi.core.configuration import AppSettings, validate_runtime_settings
from vidi.core.types import Detection, DetectionResult, FramePacket

logger = logging.getLogger(__name__)
WINDOW_TITLE = "vidi (q/Esc to exit)"


class SimulationError(RuntimeError):
    """A requested local preview, image export, or report failed."""


@dataclass(frozen=True)
class SimulationResult:
    """Measured application results; rates are not sensor accuracy guarantees."""

    frames_processed: int
    width: int
    height: int
    output_path: Path | None
    backend: str = "none"
    simulated: bool = False
    ready_frames: int = 0
    not_ready_frames: int = 0
    detections: tuple[Detection, ...] = ()
    startup_s: float = 0.0
    processing_s: float = 0.0
    report_path: Path | None = None
    description: str | None = None
    description_status: str = "not_requested"
    description_error: str | None = None
    description_model: str | None = None
    description_elapsed_s: float | None = None


def _open_source(settings: AppSettings):
    if settings.camera.source == "imx500":
        from vidi.camera.imx500 import Imx500Capture

        return Imx500Capture(settings.camera, settings.models, settings.detection)
    return FrameCapture(settings.camera)


def _save_image(path: Path, image) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise SimulationError(f"OpenCV could not save the frame to {path}.")
    except (OSError, cv2.error) as exc:
        raise SimulationError(f"Cannot save frame to {path}: {exc}") from exc


def run_simulation(
    settings: AppSettings, *, describe: bool = False, question: str = "What is in front of me?"
) -> SimulationResult:
    """Run selected backends; only a per-run describe argument permits an upload.

    A configured backend or API key never initiates a request. The optional
    single image submission happens after local camera resources are released.
    """
    validate_runtime_settings(settings, describe=describe)
    backend = settings.models.object_detection_backend or "none"
    preview = settings.simulation.preview
    if preview and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise SimulationError("Preview needs a desktop display. Run with --no-preview instead.")

    detector = None
    if backend == "fixture":
        from vidi.ai.object_detection import FixtureDetector

        detector = FixtureDetector(settings.detection.fixture_path, settings.detection)

    count = ready = not_ready = 0
    last_packet = last_detection = None
    first_frame_ns = first_ready_ns = last_ready_ns = None
    records = []
    startup_started = time.monotonic()
    processing_started = startup_started
    try:
        with _open_source(settings) as capture:
            processing_started = time.monotonic()
            for _ in range(settings.simulation.max_frames):
                started = time.monotonic()
                if settings.camera.source == "imx500":
                    packet = capture.read_packet()
                else:
                    frame = capture.read()
                    packet = None if frame is None else FramePacket(count + 1, time.monotonic_ns(), frame)
                if packet is None:
                    raise CaptureError(
                        f"Frame source stopped delivering frames after {count} frame(s). "
                        "Check the camera or run with --source sample."
                    )
                count += 1
                if first_frame_ns is None:
                    first_frame_ns = packet.captured_monotonic_ns
                result = detector.detect(packet) if detector is not None else packet.inference
                if backend == "imx500" and result is None:
                    result = DetectionResult(packet.frame_id, backend, "not_ready")
                if result is not None and result.frame_id != packet.frame_id:
                    from vidi.ai.object_detection import DetectionError

                    raise DetectionError("Inference metadata belongs to a different frame.")
                if result is not None:
                    ready += result.status == "ready"
                    not_ready += result.status == "not_ready"
                    if result.status == "ready":
                        if first_ready_ns is None:
                            first_ready_ns = packet.captured_monotonic_ns
                        last_ready_ns = packet.captured_monotonic_ns
                last_packet, last_detection = packet, result
                if settings.simulation.report_path is not None:
                    records.append({
                        "frame_id": packet.frame_id,
                        "captured_monotonic_ns": packet.captured_monotonic_ns,
                        "width": packet.image_bgr.shape[1],
                        "height": packet.image_bgr.shape[0],
                        "inference": asdict(result) if result is not None else None,
                    })
                logger.debug("Frame %d: %dx%d", count, packet.image_bgr.shape[1], packet.image_bgr.shape[0])
                if preview:
                    display = packet.image_bgr
                    if result is not None:
                        from vidi.ai.object_detection import render_detections

                        display = render_detections(display, result)
                    cv2.imshow(WINDOW_TITLE, display)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if count < settings.simulation.max_frames:
                    time.sleep(max(0.0, 1.0 / settings.camera.fps - (time.monotonic() - started)))
            processing_ended = time.monotonic()
    except cv2.error as exc:
        raise SimulationError(f"OpenCV preview failed; try --no-preview. Details: {exc}") from exc
    finally:
        if preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                logger.debug("OpenCV had no preview windows to close.")

    processing_s = processing_ended - processing_started
    startup_s = processing_started - startup_started
    assert last_packet is not None
    output = settings.simulation.output_path
    if output is not None:
        image = last_packet.image_bgr
        if last_detection is not None:
            from vidi.ai.object_detection import render_detections

            image = render_detections(image, last_detection)
        _save_image(output, image)

    description = description_error = description_model = description_elapsed_s = None
    description_status = "not_requested"
    if describe:
        from vidi.ai.scene_description import GeminiSceneDescriber, SceneDescriptionError

        try:
            response = GeminiSceneDescriber(settings.cloud).describe(last_packet.image_bgr, question)
            description = response.text
            description_model = response.model
            description_elapsed_s = response.elapsed_s
            description_status = "ready"
            logger.info("Gemini description: %s", description)
        except SceneDescriptionError as exc:
            description_status = "unavailable"
            description_error = str(exc)
            logger.warning("Gemini unavailable: %s. Local results are preserved.", exc)

    result = SimulationResult(
        count, last_packet.image_bgr.shape[1], last_packet.image_bgr.shape[0], output,
        backend=backend, simulated=backend == "fixture", ready_frames=ready,
        not_ready_frames=not_ready,
        detections=last_detection.detections if last_detection is not None else (),
        startup_s=startup_s, processing_s=processing_s,
        report_path=settings.simulation.report_path, description=description,
        description_status=description_status, description_error=description_error,
        description_model=description_model, description_elapsed_s=description_elapsed_s,
    )
    if result.report_path is not None:
        report = asdict(result)
        report["output_path"] = str(output) if output is not None else None
        report["report_path"] = str(result.report_path)
        report.update({
            "schema_version": 1, "source": settings.camera.source, "frames": records,
            "capture_fps": _observed_rate(count, first_frame_ns, last_packet.captured_monotonic_ns),
            "inference_ready_fps": _observed_rate(ready, first_ready_ns, last_ready_ns),
            "last_inference_status": last_detection.status if last_detection is not None else "disabled",
        })
        try:
            result.report_path.parent.mkdir(parents=True, exist_ok=True)
            result.report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            raise SimulationError(f"Cannot save JSON report to {result.report_path}: {exc}") from exc

    if backend == "none":
        logger.info(
            "Simulation completed: %d frame(s), %dx%d, source=%s. %s",
            count, result.width, result.height, settings.camera.source,
            "AI and audio are not active." if not describe else "Local detection and audio are not active.",
        )
    else:
        logger.info(
            "%s detection completed: %d frame(s), %d ready, %d not ready; last frame has %d detection(s).",
            "SIMULATED" if result.simulated else "Local AI Camera",
            count, ready, not_ready, len(result.detections),
        )
    if output is not None:
        logger.info("Last frame saved to %s", output)
    if result.report_path is not None:
        logger.info("Run report saved to %s", result.report_path)
    return result


def _observed_rate(count: int, first_ns: int | None, last_ns: int | None) -> float | None:
    """Use delivery intervals; one frame provides no observable frame rate."""
    if count < 2 or first_ns is None or last_ns is None or last_ns <= first_ns:
        return None
    return (count - 1) * 1_000_000_000 / (last_ns - first_ns)
