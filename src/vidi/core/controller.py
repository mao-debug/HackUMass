"""Bounded local capture/detection, reporting, and an explicit cloud opt-in."""

from __future__ import annotations

from collections import deque
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2

from vidi.camera.capture import CaptureError, FrameCapture
from vidi.core.configuration import AppSettings, validate_runtime_settings
from vidi.core.types import Detection, DetectionResult, FramePacket, OcrResult, SpeechResult

logger = logging.getLogger(__name__)
WINDOW_TITLE = "vidi (q/Esc to exit)"
CONTINUOUS_RECORD_LIMIT = 120


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
    ocr: OcrResult = field(default_factory=OcrResult)
    speech: SpeechResult = field(default_factory=SpeechResult)
    last_inference: DetectionResult | None = None
    interrupted: bool = False


@dataclass(frozen=True)
class SceneProximityResult(SimulationResult):
    """Explicit Phase 4 additions, without changing older report schemas."""

    proximity: dict | None = None
    proximity_feedback: str | None = None
    proximity_status: str = "pending"


def _open_source(settings: AppSettings):
    if settings.camera.source == "picamera2":
        from vidi.camera.picamera2 import Picamera2Capture

        return Picamera2Capture(settings.camera)
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


def _save_report(path: Path, report: dict) -> None:
    """Replace a report only after a complete, valid JSON checkpoint is ready."""
    temporary = None
    try:
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(path)
        temporary = None
    except (OSError, TypeError, ValueError) as exc:
        raise SimulationError(f"Cannot save JSON report to {path}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove a temporary JSON report file.")


def run_simulation(
    settings: AppSettings, *, describe: bool = False, question: str = "What is in front of me?",
    read_text: bool = False, speak: bool = False, speech_output: Path | None = None,
    gemini_detect: bool = False,
    continuous: bool = False,
    gemini_live: bool = False,
    gemini_read: bool = False,
    scene_live: bool = False,
    proximity: bool = False,
    voice_commands: bool = False,
    offline: bool = False,
    chess_live: bool = False,
) -> SimulationResult:
    """Run selected backends; cloud actions require an explicit per-run flag.

    A configured backend or API key never initiates a request. The optional
    single image submission happens after local camera resources are released.
    """
    validate_runtime_settings(
        settings, describe=describe, read_text=read_text, speak=speak, speech_output=speech_output,
        gemini_detect=gemini_detect, continuous=continuous, gemini_live=gemini_live, gemini_read=gemini_read,
        scene_live=scene_live, proximity=proximity,
        **({"voice_commands": True} if voice_commands else {}),
        **({"offline": True} if offline else {}),
        **({"chess_live": True} if chess_live else {}),
    )
    backend = settings.models.object_detection_backend or "none"
    preview = settings.simulation.preview
    capture_title = (
        "vidi snapshot (q/Esc finishes capture; Ctrl+C cancels)"
        if gemini_detect or describe or read_text or gemini_read else WINDOW_TITLE
    )
    if preview and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise SimulationError("Preview needs a desktop display. Run with --no-preview instead.")

    if chess_live:
        from vidi.core.chess_live import run_chess_live

        return run_chess_live(settings, speak=speak, voice_commands=voice_commands, open_source=_open_source)

    if scene_live:
        from vidi.core.phase4 import run_phase4_live

        return run_phase4_live(settings, proximity=proximity, speak=speak,
                               open_source=_open_source, question=question,
                               **({"voice_commands": True} if voice_commands else {}),
                               **({"offline": True} if offline else {}))

    detector = None
    if backend == "fixture":
        from vidi.ai.object_detection import FixtureDetector

        detector = FixtureDetector(settings.detection.fixture_path, settings.detection)
    elif backend == "opencv_ssd":
        from vidi.ai.cpu_detection import OpenCvSsdDetector

        detector = OpenCvSsdDetector(settings.models, settings.detection)

    if continuous and (settings.simulation.async_detection or gemini_live):
        from vidi.core.live import run_live_detection

        if gemini_live:
            from vidi.ai.gemini_detection import GeminiObjectDetector

            detector = GeminiObjectDetector(settings.cloud, settings.detection)
        return run_live_detection(settings, detector, open_source=_open_source, gemini_live=gemini_live)

    count = ready = not_ready = 0
    last_packet = last_detection = None
    first_frame_ns = first_ready_ns = last_ready_ns = None
    records = deque(maxlen=CONTINUOUS_RECORD_LIMIT) if continuous else []
    interrupted = False
    startup_started = time.monotonic()
    processing_started = startup_started
    try:
        with _open_source(settings) as capture:
            processing_started = time.monotonic()
            if continuous:
                logger.info(
                    "Continuous local mode active: source=%s, detector=%s. %s",
                    settings.camera.source, backend,
                    "Press q/Esc in the preview or Ctrl+C to stop." if preview else "Press Ctrl+C to stop.",
                )
            try:
                while continuous or count < settings.simulation.max_frames:
                    started = time.monotonic()
                    if settings.camera.source in ("picamera2", "imx500"):
                        packet = capture.read_packet()
                    else:
                        frame = capture.read()
                        packet = None if frame is None else FramePacket(count + 1, time.monotonic_ns(), frame)
                    if packet is None:
                        raise CaptureError(
                            f"Frame source stopped delivering frames after {count} frame(s). "
                            "Check the camera or run with --source sample."
                        )
                    result = detector.detect(packet) if detector is not None else packet.inference
                    if backend == "imx500" and result is None:
                        result = DetectionResult(packet.frame_id, backend, "not_ready")
                    if result is not None and result.frame_id != packet.frame_id:
                        from vidi.ai.object_detection import DetectionError

                        raise DetectionError("Inference metadata belongs to a different frame.")
                    # Commit only completed frame/result pairs. An interruption
                    # during inference must not mix a new count with old boxes.
                    count += 1
                    if first_frame_ns is None:
                        first_frame_ns = packet.captured_monotonic_ns
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
                        cv2.imshow(capture_title, display)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    if continuous or count < settings.simulation.max_frames:
                        time.sleep(max(0.0, 1.0 / settings.camera.fps - (time.monotonic() - started)))
            except KeyboardInterrupt:
                if not continuous or last_packet is None:
                    raise
                interrupted = True
            finally:
                processing_ended = time.monotonic()
    except KeyboardInterrupt:
        if not continuous or last_packet is None:
            raise
        interrupted = True
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
        try:
            _save_image(output, image)
        except SimulationError as exc:
            if not interrupted:
                raise
            logger.error("%s. The continuous run was interrupted.", exc)
            output = None

    description = description_error = description_model = description_elapsed_s = None
    description_status = "not_requested"
    result_type = SceneProximityResult if proximity else SimulationResult
    result = result_type(
        count, last_packet.image_bgr.shape[1], last_packet.image_bgr.shape[0], output,
        backend=backend, simulated=backend == "fixture", ready_frames=ready,
        not_ready_frames=not_ready,
        detections=last_detection.detections if last_detection is not None else (),
        startup_s=startup_s, processing_s=processing_s,
        report_path=settings.simulation.report_path, description=description,
        description_status=description_status, description_error=description_error,
        description_model=description_model, description_elapsed_s=description_elapsed_s,
        last_inference=last_detection,
        interrupted=interrupted,
    )

    report_saved = False

    def persist(current: SimulationResult) -> None:
        """Checkpoint OCR before audio starts, then publish its final outcome."""
        nonlocal report_saved
        report_saved = False
        if current.report_path is None:
            return
        report = asdict(current)
        report["output_path"] = str(output) if output is not None else None
        report["report_path"] = str(current.report_path)
        report["speech"]["output_path"] = str(current.speech.output_path) if current.speech.output_path is not None else None
        report.update({
            "schema_version": 1, "source": settings.camera.source, "frames": list(records),
            "capture_fps": _observed_rate(count, first_frame_ns, last_packet.captured_monotonic_ns),
            "inference_ready_fps": _observed_rate(ready, first_ready_ns, last_ready_ns),
            "last_inference_status": last_detection.status if last_detection is not None else "disabled",
        })
        if proximity:
            report.update({"schema_version": 3, "mode": "phase4_scene_snapshot"})
        if continuous:
            report.update({
                "continuous": True,
                "frame_records_limit": CONTINUOUS_RECORD_LIMIT,
                "frame_records_truncated": count > len(records),
                "frame_records_dropped": count - len(records),
            })
        _save_report(current.report_path, report)
        report_saved = True

    def persist_interruption(current: SimulationResult) -> None:
        """A checkpoint failure must not replace the user's Ctrl+C outcome."""
        try:
            persist(current)
        except SimulationError as exc:
            logger.error("%s. The interrupted run retains its previous checkpoint, if available.", exc)

    speech_input = None
    speech_skip = "skipped_cloud_failed"

    if describe:
        from vidi.ai.scene_description import GeminiSceneDescriber, SceneDescriptionError

        result = replace(result, description_status="pending", description_model=settings.cloud.model)
        persist(result)
        description_started = time.monotonic()
        try:
            response = GeminiSceneDescriber(settings.cloud).describe(last_packet.image_bgr, question)
            result = replace(
                result, description=response.text, description_model=response.model,
                description_elapsed_s=response.elapsed_s, description_status="ready",
            )
            speech_input = response.text
            logger.info("Gemini description: %s", response.text)
        except SceneDescriptionError as exc:
            result = replace(
                result, description_status="unavailable", description_error=str(exc),
                description_elapsed_s=time.monotonic() - description_started,
            )
            logger.warning("Gemini unavailable: %s. Local results are preserved.", exc)
        except KeyboardInterrupt:
            result = replace(
                result, description_status="cancelled",
                description_elapsed_s=time.monotonic() - description_started,
            )
            persist_interruption(result)
            raise
        persist(result)

    if gemini_detect:
        from vidi.ai.gemini_detection import GeminiObjectDetector
        from vidi.ai.object_detection import DetectionError, render_detections

        def record_cloud_result(inference: DetectionResult) -> None:
            nonlocal result, last_detection, ready, first_ready_ns, last_ready_ns
            if inference.frame_id != last_packet.frame_id:
                raise DetectionError("Gemini inference belongs to a different snapshot.")
            last_detection = inference
            ready = int(inference.status == "ready")
            if ready:
                first_ready_ns = last_ready_ns = last_packet.captured_monotonic_ns
            if records:
                records[-1]["inference"] = asdict(inference)
            result = replace(
                result, ready_frames=ready, not_ready_frames=0,
                detections=inference.detections, last_inference=inference,
            )

        record_cloud_result(DetectionResult(
            last_packet.frame_id, "gemini", "pending", model=settings.cloud.model,
        ))
        persist(result)
        cloud_started = time.monotonic()
        try:
            inference = GeminiObjectDetector(settings.cloud, settings.detection).detect(last_packet)
        except KeyboardInterrupt:
            record_cloud_result(DetectionResult(
                last_packet.frame_id, "gemini", "cancelled", model=settings.cloud.model,
                elapsed_s=time.monotonic() - cloud_started,
            ))
            persist_interruption(result)
            raise
        record_cloud_result(inference)
        persist(result)
        annotated = render_detections(last_packet.image_bgr, inference)
        if output is not None:
            _save_image(output, annotated)
        if preview:
            try:
                cv2.imshow("vidi Gemini result", annotated)
                cv2.waitKey(1000)
            except cv2.error as exc:
                raise SimulationError("Cannot preview the Gemini snapshot; try --no-preview.") from exc
            finally:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
        if inference.status == "ready":
            speech_input = (
                "Objects identified: " + ", ".join(item.label for item in inference.detections) + "."
                if inference.detections else "No objects identified in this snapshot."
            )
            print(speech_input, flush=True)
        else:
            logger.error("Gemini detection unavailable: %s", inference.error or "No usable result.")

    if read_text or gemini_read:
        ocr_backend = "gemini" if gemini_read else "tesseract"
        ocr_model = settings.cloud.model if gemini_read else None
        if gemini_read:
            from vidi.ai.gemini_text_recognition import GeminiTextRecognizer

            recognizer = GeminiTextRecognizer(settings.cloud, settings.ocr)
            result = replace(result, ocr=OcrResult(
                status="pending", frame_id=last_packet.frame_id, backend=ocr_backend, model=ocr_model,
            ))
            persist(result)
            logger.info("Gemini text reading requested; waiting for one response (request timeout %.1fs).", settings.ocr.timeout_s)
        else:
            from vidi.ai.text_recognition import TesseractRecognizer

            recognizer = TesseractRecognizer(settings.ocr)

        ocr_started = time.monotonic()
        try:
            ocr = recognizer.recognize(last_packet)
        except KeyboardInterrupt:
            result = replace(result, ocr=OcrResult(
                status="cancelled", frame_id=last_packet.frame_id, backend=ocr_backend, model=ocr_model,
                elapsed_s=time.monotonic() - ocr_started,
            ))
            persist_interruption(result)
            raise
        speech_input = ocr.text if ocr.status in ("ready", "no_text") else None
        speech_skip = "skipped_ocr_failed"
        result = replace(result, ocr=ocr)
        persist(result)
        # OCR text is user-requested output, not a background diagnostic log.
        if ocr.status == "ready":
            print(ocr.text, flush=True)
        elif ocr.status == "no_text":
            print("No readable text found.", flush=True)
        else:
            logger.error("Text reading failed: %s", ocr.error or "OCR unavailable.")
    sensor_snapshot = None
    if proximity:
        from vidi.core.phase4 import compose_feedback
        from vidi.sensors.proximity import (
            ProximityMonitor, format_proximity, refresh_snapshot, unknown_snapshot,
        )

        checkpoint_ns = time.monotonic_ns()
        try:
            with ProximityMonitor(settings.proximity) as monitor:
                sensor_snapshot = monitor.fresh(
                    after_ns=checkpoint_ns,
                    timeout_s=settings.proximity.startup_timeout_s + 0.75,
                )
        except KeyboardInterrupt:
            result = replace(result, proximity_status="cancelled")
            persist_interruption(result)
            raise
        except Exception:
            sensor_snapshot = unknown_snapshot(error="Local proximity is unavailable.")
        # Shutdown and checkpoint I/O can consume the freshness budget too.
        sensor_snapshot = refresh_snapshot(sensor_snapshot, settings.proximity.stale_after_s)
        proximity_feedback = format_proximity(sensor_snapshot, settings.proximity.nearby_mm)
        result = replace(
            result, proximity=sensor_snapshot.to_dict(), proximity_feedback=proximity_feedback,
            proximity_status="ready" if any(item.status == "ok" for item in sensor_snapshot.observations) else "unavailable",
        )
        print(proximity_feedback, flush=True)
        persist(result)

    requested_speech = speak or speech_output is not None
    if requested_speech:
        if proximity:
            sensor_snapshot = refresh_snapshot(sensor_snapshot, settings.proximity.stale_after_s)
            proximity_feedback = format_proximity(sensor_snapshot, settings.proximity.nearby_mm)
            if result.description_status != "ready":
                proximity_feedback += " Scene description unavailable."
            speech_input = compose_feedback(
                result.description, proximity_feedback,
                max(0.0, (time.monotonic_ns() - last_packet.captured_monotonic_ns) / 1e9),
                settings.speech.max_words,
            )
            result = replace(
                result, proximity=sensor_snapshot.to_dict(), proximity_feedback=proximity_feedback,
                proximity_status="ready" if any(item.status == "ok" for item in sensor_snapshot.observations) else "unavailable",
            )
        pending = SpeechResult(
            status="pending" if speech_input is not None else speech_skip,
            mode="wav" if speech_output is not None else "playback", output_path=speech_output,
        )
        result = replace(result, speech=pending)
        persist(result)
        if speech_input is not None:
            if proximity:
                # A slow checkpoint must not turn a formerly fresh return into
                # an apparently current observation at the audio boundary.
                sensor_snapshot = refresh_snapshot(sensor_snapshot, settings.proximity.stale_after_s)
                proximity_feedback = format_proximity(sensor_snapshot, settings.proximity.nearby_mm)
                if result.description_status != "ready":
                    proximity_feedback += " Scene description unavailable."
                speech_input = compose_feedback(
                    result.description, proximity_feedback,
                    max(0.0, (time.monotonic_ns() - last_packet.captured_monotonic_ns) / 1e9),
                    settings.speech.max_words,
                )
                result = replace(
                    result, proximity=sensor_snapshot.to_dict(), proximity_feedback=proximity_feedback,
                    proximity_status="ready" if any(item.status == "ok" for item in sensor_snapshot.observations) else "unavailable",
                )
            from vidi.audio.text_to_speech import LocalSpeech

            speech_started = time.monotonic()
            try:
                spoken = LocalSpeech(settings.speech, settings.models.text_to_speech_backend).speak(
                    speech_input, output_path=speech_output,
                )
            except KeyboardInterrupt:
                result = replace(result, speech=replace(
                    pending, status="cancelled", elapsed_s=time.monotonic() - speech_started,
                ))
                persist_interruption(result)
                raise
            result = replace(result, speech=spoken)
            if spoken.status == "failed":
                logger.error("Speech failed: %s. Snapshot results are preserved.", spoken.error)
            elif spoken.mode == "wav":
                logger.info("Speech synthesized to %s; playback was not requested.", spoken.output_path)
            else:
                logger.info("Speech engine completed; physical audibility needs a connected output device.")

    if interrupted:
        persist_interruption(result)
    else:
        persist(result)

    if read_text or gemini_read:
        logger.info("%s text reading completed: OCR=%s, speech=%s.",
                    "Gemini" if gemini_read else "Local", result.ocr.status, result.speech.status)
    elif backend == "none":
        logger.info(
            "Simulation completed: %d frame(s), %dx%d, source=%s. %s",
            count, result.width, result.height, settings.camera.source,
            "AI and audio are not active." if not describe else (
                f"Gemini description={result.description_status}; speech={result.speech.status}. Local detection is inactive."
            ),
        )
    else:
        logger.info(
            "%s detection completed: %d frame(s), %d ready, %d not ready; last frame has %d detection(s).",
            "SIMULATED" if result.simulated else ("Gemini snapshot" if gemini_detect else f"Local {backend}"),
            count, ready, not_ready, len(result.detections),
        )
    if output is not None:
        logger.info("Last frame saved to %s", output)
    if result.report_path is not None and report_saved:
        logger.info("Run report saved to %s", result.report_path)
    return result


def _observed_rate(count: int, first_ns: int | None, last_ns: int | None) -> float | None:
    """Use delivery intervals; one frame provides no observable frame rate."""
    if count < 2 or first_ns is None or last_ns is None or last_ns <= first_ns:
        return None
    return (count - 1) * 1_000_000_000 / (last_ns - first_ns)
