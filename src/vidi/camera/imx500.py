"""Opt-in Raspberry Pi AI Camera capture with bounded worker-process waits.

Picamera2 is imported only in the spawned hardware worker. Each result is decoded
from the same completed request as its image before that request is released.
Only the official SSD MobileNetV2 FPNLite 320 PP model contract is supported.

API references verified against Raspberry Pi's Picamera2 request/IMX500 sources
and picamera2-examples/examples/imx500/imx500_object_detection_demo.py.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
from multiprocessing import shared_memory
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from vidi.ai.object_detection import DetectionError, filter_detections, load_labels, supported_label
from vidi.camera.capture import CaptureError
from vidi.camera.managed import ManagedPiCapture, _JOIN_TIMEOUT_S, _RESULT_BYTES, _packet_metadata, run_worker
from vidi.core.types import Detection, DetectionResult, FramePacket

if TYPE_CHECKING:
    from vidi.core.configuration import CameraSettings, DetectionSettings, ModelSettings

logger = logging.getLogger(__name__)
SUPPORTED_MODEL = "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"


def _load_pi_dependencies():
    """Leave macOS imports and all simulation backends independent of Pi APIs."""
    try:
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
    except ImportError as exc:
        raise CaptureError(
            "AI Camera capture needs Raspberry Pi OS Picamera2/libcamera and IMX500 firmware. "
            "Use --source sample on a development computer."
        ) from exc
    return Picamera2, IMX500


class _PiSession:
    """Worker-local camera owner; no hardware objects cross process boundaries."""

    def __init__(self, camera, models, detection) -> None:
        self.camera = None
        self.sensor = None
        self.detection = detection
        self.enabled = models.object_detection_backend == "imx500"
        self.labels: tuple[str, ...] = ()
        self.intrinsics = None
        Picamera2, IMX500 = _load_pi_dependencies()
        try:
            # The IMX500 helper must be constructed before Picamera2. Empty means
            # camera-only capture, with no network model firmware selected.
            self.sensor = IMX500(str(models.object_detection_path) if self.enabled else "")
            if self.enabled:
                self.intrinsics = self.sensor.network_intrinsics
                task = getattr(self.intrinsics, "task", None)
                postprocess = getattr(self.intrinsics, "postprocess", None)
                if task not in (None, "object detection") or postprocess not in (None, ""):
                    raise DetectionError("This decoder supports the baseline SSD object-detection outputs only.")
                if tuple(self.sensor.get_input_size()) != (320, 320):
                    raise DetectionError("The supported SSD model requires a 320x320 input tensor.")
                if detection.labels_path:
                    self.labels = load_labels(detection.labels_path)
                else:
                    embedded = getattr(self.intrinsics, "labels", None)
                    if not embedded or any(not isinstance(label, str) or len(label) > 256 for label in embedded):
                        raise DetectionError(
                            "The model has no usable embedded labels. Set detection.labels_path "
                            "to a class-indexed labels file, retaining blank and '-' lines."
                        )
                    self.labels = tuple(embedded)
            self.camera = Picamera2(self.sensor.camera_num)
            config = self.camera.create_preview_configuration(
                main={"size": (camera.width, camera.height), "format": "RGB888"},
                controls={"FrameRate": camera.fps}, buffer_count=4, queue=False,
            )
            # Picamera2's RGB888 array bytes are B,G,R (its documented libcamera
            # format convention), matching the rest of this application's frames.
            self.camera.configure(config)
            self.camera.start(show_preview=False)
            if self.enabled and getattr(self.intrinsics, "preserve_aspect_ratio", False):
                self.sensor.set_auto_aspect_ratio()
        except BaseException:
            self.close()
            raise

    def _decode(self, metadata: dict, frame_id: int, width: int, height: int) -> DetectionResult:
        outputs = self.sensor.get_outputs(metadata, add_batch=True)
        if outputs is None:
            return DetectionResult(frame_id, "imx500", "not_ready")
        if not isinstance(outputs, (list, tuple)) or len(outputs) not in (3, 4):
            raise DetectionError("Unsupported SSD output tensors; expected boxes, scores, classes and optional count.")
        boxes, scores, classes = (np.asarray(value) for value in outputs[:3])
        if (
            boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4
            or scores.shape != boxes.shape[:2] or classes.shape != scores.shape
        ):
            raise DetectionError("Unsupported SSD tensor shapes; expected [1,N,4], [1,N], [1,N].")
        boxes, scores, classes = boxes[0], scores[0], classes[0]
        count = len(scores)
        if len(outputs) == 4:
            raw_count = np.asarray(outputs[3])
            if raw_count.size != 1 or not np.isfinite(raw_count).all():
                raise DetectionError("Invalid SSD detection count tensor.")
            value = float(raw_count.reshape(-1)[0])
            if value != int(value) or not 0 <= value <= count:
                raise DetectionError("Invalid SSD detection count.")
            count = int(value)
        boxes, scores, classes = boxes[:count], scores[:count], classes[:count]
        if not all(np.isfinite(array).all() for array in (boxes, scores, classes)):
            raise DetectionError("SSD output contains non-finite values.")
        if ((scores < 0) | (scores > 1)).any() or (classes != np.floor(classes)).any():
            raise DetectionError("SSD output has invalid confidence or class indices.")
        if getattr(self.intrinsics, "bbox_normalization", False):
            boxes = boxes / 320.0
        order = getattr(self.intrinsics, "bbox_order", None) or "yx"
        if order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        elif order != "yx":
            raise DetectionError("Unsupported SSD bounding-box coordinate order.")
        detections = []
        for box, score, category in zip(boxes, scores, classes):
            class_id = int(category)
            if (
                score < self.detection.confidence_threshold or not 0 <= class_id < len(self.labels)
                or not supported_label(self.labels[class_id])
            ):
                continue
            y0, x0, y1, x1 = np.clip(box, 0.0, 1.0)
            if y1 <= y0 or x1 <= x0:
                raise DetectionError("SSD output contains an empty or reversed bounding box.")
            # The helper accounts for this request's ScalerCrop/inference ROI.
            mapped = self.sensor.convert_inference_coords((y0, x0, y1, x1), metadata, self.camera)
            if len(mapped) != 4 or not all(math.isfinite(float(value)) for value in mapped):
                raise DetectionError("IMX500 coordinate conversion returned invalid image coordinates.")
            x, y, box_width, box_height = mapped
            left, top = max(0, int(x)), max(0, int(y))
            right, bottom = min(width, int(x + box_width)), min(height, int(y + box_height))
            if left >= right or top >= bottom:
                continue
            detections.append(Detection(class_id, self.labels[class_id].strip(), float(score), (left, top, right, bottom)))
        return DetectionResult(frame_id, "imx500", "ready", filter_detections(detections, self.detection))

    def read_packet(self, frame_id: int) -> FramePacket | None:
        request = self.camera.capture_request(flush=True)
        if request is None:
            return None
        try:
            image = np.array(request.make_array("main"), copy=True, order="C")
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise CaptureError("AI Camera did not provide the configured uint8 BGR image.")
            captured_at = time.monotonic_ns()
            metadata = request.get_metadata()
            inference = self._decode(metadata, frame_id, image.shape[1], image.shape[0]) if self.enabled else None
            return FramePacket(frame_id, captured_at, image, inference)
        finally:
            request.release()

    def close(self) -> None:
        camera, self.camera = self.camera, None
        sensor, self.sensor = self.sensor, None
        if camera is not None:
            for action in (camera.stop, camera.close):
                try:
                    action()
                except Exception as exc:
                    logger.warning("AI Camera cleanup could not complete a camera operation: %s", exc)
        device_fd = getattr(sensor, "device_fd", None)
        if device_fd is not None:
            try:
                device_fd.close()
            except Exception as exc:
                logger.warning("AI Camera cleanup could not close the sensor descriptor: %s", exc)


def _worker_main(connection, shared_name: str, frame_bytes: int, camera, models, detection) -> None:
    """Keep IMX500 ownership inside its dedicated spawned worker."""
    run_worker(connection, shared_name, frame_bytes, _PiSession, (camera, models, detection))


class Imx500Capture(ManagedPiCapture):
    """Capture AI Camera frames with the optional supported on-sensor detector."""

    def __init__(self, camera_settings: CameraSettings, model_settings: ModelSettings,
                 detection_settings: DetectionSettings) -> None:
        self.model_settings = model_settings
        self.detection_settings = detection_settings
        super().__init__(
            camera_settings, _worker_main, (camera_settings, model_settings, detection_settings),
            source_name="AI Camera", startup_stage="startup/model transfer",
            startup_message="Starting AI Camera; local model firmware transfer may take several minutes.",
            ready_message="AI Camera worker ready; missing inference outputs will be reported as not_ready.",
            accept_inference=True,
        )

    def _validate_source(self) -> None:
        backend = self.model_settings.object_detection_backend
        if backend not in (None, "none", "imx500"):
            raise DetectionError("The AI Camera source supports the imx500 or none detector backend.")
        if backend == "imx500":
            path = self.model_settings.object_detection_path
            if path is None or not path.is_file():
                raise DetectionError(f"Local IMX500 model file does not exist: {path}")
            if path.name != SUPPORTED_MODEL:
                raise DetectionError(f"This decoder supports {SUPPORTED_MODEL} only; other models need their own decoder.")
