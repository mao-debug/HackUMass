"""Responsive capture with a separate, accurately labeled analyzed-frame view."""

from collections import deque
from contextlib import nullcontext
from dataclasses import asdict
import logging
import time

import cv2
import numpy as np

from vidi.ai.live_detection import LiveDetector
from vidi.ai.object_detection import render_detections
from vidi.camera.capture import CaptureError
from vidi.core.configuration import AppSettings, validate_runtime_settings
from vidi.core.controller import (
    CONTINUOUS_RECORD_LIMIT, SimulationError, SimulationResult,
    _observed_rate, _save_image, _save_report,
)
from vidi.core.types import FramePacket

logger = logging.getLogger(__name__)
WINDOW_TITLE = "vidi live camera | last analyzed frame (q/Esc to quit)"


def _caption(image, text: str, *, bottom: bool = False) -> None:
    height, width = image.shape[:2]
    top = max(0, height - 30) if bottom else 0
    cv2.rectangle(image, (0, top), (width, min(height, top + 30)), (25, 25, 25), -1)
    cv2.putText(image, text, (5, min(height - 1, top + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                min(0.5, max(0.15, width / 1200)), (245, 245, 245), 1, cv2.LINE_AA)


def _notice(image, lines: tuple[str, ...]) -> None:
    """Make request outcomes visible without implying a successful detection."""
    height, width = image.shape[:2]
    spacing = min(30, max(12, height // 8))
    top = max(30, (height - spacing * len(lines)) // 2)
    cv2.rectangle(image, (0, top - 5), (width, min(height, top + spacing * len(lines))), (25, 25, 25), -1)
    for index, text in enumerate(lines):
        cv2.putText(image, text, (10, top + spacing * index + spacing - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    min(0.55, max(0.15, width / 1100)), (245, 245, 245), 1, cv2.LINE_AA)


def render_live_preview(packet: FramePacket, analyzed, *, paused: bool = False, detection_enabled: bool = True,
                        request_timeout_s: float | None = None):
    """Never attach an older prediction to the current live-camera image."""
    live = packet.image_bgr.copy()
    _caption(live, f"LIVE CAMERA - frame {packet.frame_id}")
    if analyzed is None:
        analysis = np.zeros_like(live)
        _caption(analysis, "WAITING FOR ANALYSIS" if detection_enabled else "DETECTOR OFF")
        if detection_enabled:
            lines = (
                ("Gemini request pending", f"Request timeout: {request_timeout_s:g}s", "Live camera continues")
                if request_timeout_s is not None else ("Waiting for detection results",)
            )
            _notice(analysis, lines)
    else:
        old_packet, inference = analyzed
        analysis = render_detections(old_packet.image_bgr, inference)
        age = max(0.0, (time.monotonic_ns() - old_packet.captured_monotonic_ns) / 1e9)
        status = " - UPLOADS PAUSED" if paused else ""
        if inference.status != "ready":
            message = (
                "Gemini request timed out" if "timed out" in (inference.error or "").lower()
                else "Detection unavailable"
            )
            _notice(analysis, (message, "No detection result received", "Uploads paused" if paused else "See terminal for details"))
        elif not inference.detections:
            _notice(analysis, ("No objects identified", "in this analyzed frame"))
        action = "ANALYZED" if inference.status == "ready" else "ATTEMPTED"
        _caption(analysis, f"{action} frame {old_packet.frame_id} - {age:.1f}s old{status}", bottom=True)
        if analysis.shape != live.shape:
            analysis = cv2.resize(analysis, (live.shape[1], live.shape[0]))
    return cv2.hconcat((live, analysis))


def run_live_detection(settings: AppSettings, detector, *, open_source, gemini_live: bool = False) -> SimulationResult:
    """Keep capture responsive; submit only bounded, replaceable worker input.

    Cloud mode needs its own explicit opt-in. Its cooldown is measured from the
    end of a request and any non-ready result stops further submissions.
    """
    validate_runtime_settings(settings, continuous=True, gemini_live=gemini_live)
    backend = settings.models.object_detection_backend or "none"
    preview = settings.simulation.preview
    worker_context = (
        LiveDetector(detector, min_interval_s=settings.cloud.live_interval_s if gemini_live else 0.0,
                     stop_on_unavailable=gemini_live)
        if detector is not None else nullcontext(None)
    )
    count = ready = not_ready = completed = 0
    last_packet = analyzed = None
    first_frame_ns = first_ready_ns = last_ready_ns = None
    last_successful_frame_id = None
    records = deque(maxlen=CONTINUOUS_RECORD_LIMIT)
    interrupted = paused = False
    started = processing_started = processing_ended = time.monotonic()
    try:
        with worker_context as worker:
            with open_source(settings) as capture:
                processing_started = time.monotonic()
                logger.info("Live camera active: requested %.1f FPS; %s identification runs separately. Press q/Esc or Ctrl+C to quit.",
                            settings.camera.fps, backend)
                if gemini_live:
                    logger.info("Repeated Gemini uploads enabled: one request at a time, at least %.1fs between completed requests and the next request; stop on error.",
                                settings.cloud.live_interval_s)
                try:
                    while True:
                        frame_started = time.monotonic()
                        if settings.camera.source in ("picamera2", "imx500"):
                            packet = capture.read_packet()
                        else:
                            frame = capture.read()
                            packet = None if frame is None else FramePacket(count + 1, time.monotonic_ns(), frame)
                        if packet is None:
                            raise CaptureError("The live camera stopped delivering frames. Check the connection.")
                        count += 1
                        last_packet = packet
                        if first_frame_ns is None:
                            first_frame_ns = packet.captured_monotonic_ns
                        if worker is not None:
                            received = worker.poll()
                            if received is not None:
                                analyzed = received
                                old_packet, inference = received
                                completed += 1
                                ready += inference.status == "ready"
                                not_ready += inference.status != "ready"
                                if inference.status == "ready":
                                    last_successful_frame_id = old_packet.frame_id
                                    received_ns = time.monotonic_ns()
                                    if first_ready_ns is None:
                                        first_ready_ns = received_ns
                                    last_ready_ns = received_ns
                                    logger.info("Analyzed frame %d: %s", old_packet.frame_id,
                                                ", ".join(item.label for item in inference.detections) or "no objects identified")
                                elif gemini_live:
                                    paused = True
                                    logger.error("Gemini uploads stopped: %s. The camera continues; restart the command to try again.",
                                                 (inference.error or "No usable result").rstrip(". "))
                                if settings.simulation.report_path is not None:
                                    records.append({
                                        "frame_id": old_packet.frame_id,
                                        "captured_monotonic_ns": old_packet.captured_monotonic_ns,
                                        "width": old_packet.image_bgr.shape[1], "height": old_packet.image_bgr.shape[0],
                                        "inference": asdict(inference),
                                    })
                            if not worker.stopped:
                                worker.submit(packet)
                        if preview:
                            cv2.imshow(WINDOW_TITLE, render_live_preview(
                                packet, analyzed, paused=paused, detection_enabled=detector is not None,
                                request_timeout_s=settings.cloud.timeout_s if gemini_live else None,
                            ))
                            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                                break
                        time.sleep(max(0.0, 1.0 / settings.camera.fps - (time.monotonic() - frame_started)))
                finally:
                    processing_ended = time.monotonic()
                    # Discard pending uploads before camera teardown can wait.
                    if worker is not None:
                        worker.close()
    except KeyboardInterrupt:
        if last_packet is None:
            raise
        interrupted = True
    except cv2.error as exc:
        raise SimulationError("Live preview failed; try --no-preview.") from exc
    finally:
        if preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    assert last_packet is not None
    inference = analyzed[1] if analyzed is not None else None
    export_packet = analyzed[0] if analyzed is not None else last_packet
    output = settings.simulation.output_path
    if output is not None:
        image = render_detections(export_packet.image_bgr, inference) if inference is not None else export_packet.image_bgr
        try:
            _save_image(output, image)
        except SimulationError as exc:
            if not interrupted:
                raise
            logger.error("%s", exc)
            output = None
    result = SimulationResult(
        count, last_packet.image_bgr.shape[1], last_packet.image_bgr.shape[0], output,
        backend=backend, simulated=backend == "fixture", ready_frames=ready, not_ready_frames=not_ready,
        detections=inference.detections if inference is not None else (),
        startup_s=processing_started - started, processing_s=processing_ended - processing_started,
        report_path=settings.simulation.report_path, last_inference=inference, interrupted=interrupted,
    )
    observed_fps = _observed_rate(count, first_frame_ns, last_packet.captured_monotonic_ns)
    if result.report_path is not None:
        report = asdict(result)
        report.update({
            "schema_version": 2, "mode": "continuous_async", "continuous": True,
            "source": settings.camera.source, "output_path": str(output) if output is not None else None,
            "report_path": str(result.report_path), "output_frame_id": export_packet.frame_id if output is not None else None,
            "last_preview_frame_id": last_packet.frame_id,
            "last_analyzed_frame_id": last_successful_frame_id,
            "last_result_frame_id": inference.frame_id if inference is not None else None,
            "last_inference_status": inference.status if inference is not None else "not_received",
            "capture_fps": observed_fps,
            "inference_ready_fps": _observed_rate(ready, first_ready_ns, last_ready_ns),
            "inference_results_received": completed, "frames": list(records),
            "frame_records_scope": "received_inference_results_only", "frame_records_limit": CONTINUOUS_RECORD_LIMIT,
            "frame_records_truncated": completed > len(records), "frame_records_dropped": completed - len(records),
            "cloud_uploads_paused": paused,
            "cloud_min_interval_s": settings.cloud.live_interval_s if gemini_live else None,
            "cloud_timeout_s": settings.cloud.timeout_s if gemini_live else None,
        })
        try:
            _save_report(result.report_path, report)
        except SimulationError as exc:
            if not interrupted:
                raise
            logger.error("%s", exc)
    logger.info("Live capture finished: %d camera frames, %d successful analyses, %d unavailable results; measured capture rate=%s FPS.",
                count, ready, not_ready, f"{observed_fps:.2f}" if observed_fps is not None else "not measured")
    return result
