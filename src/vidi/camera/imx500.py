"""Opt-in Raspberry Pi AI Camera capture with bounded worker-process waits.

Picamera2 is imported only in the spawned hardware worker. Each result is decoded
from the same completed request as its image before that request is released.
Only the official SSD MobileNetV2 FPNLite 320 PP model contract is supported.

API references verified against Raspberry Pi's Picamera2 request/IMX500 sources
and picamera2-examples/examples/imx500/imx500_object_detection_demo.py.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from vidi.ai.object_detection import DetectionError, filter_detections, load_labels, supported_label
from vidi.camera.capture import CaptureError
from vidi.core.types import Detection, DetectionResult, FramePacket

if TYPE_CHECKING:
    from vidi.core.configuration import CameraSettings, DetectionSettings, ModelSettings

logger = logging.getLogger(__name__)
SUPPORTED_MODEL = "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
_RESULT_BYTES = 65536
_JOIN_TIMEOUT_S = 1.0


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


def _packet_metadata(packet: FramePacket) -> bytes:
    result = packet.inference
    if result is not None and result.frame_id != packet.frame_id:
        raise DetectionError("Inference result is associated with a different camera frame.")
    data = {
        "frame_id": packet.frame_id, "captured_monotonic_ns": packet.captured_monotonic_ns,
        "shape": list(packet.image_bgr.shape), "inference": None,
    }
    if result is not None:
        data["inference"] = {
            "backend": result.backend, "status": result.status, "simulated": result.simulated,
            "detections": [
                {"class_id": item.class_id, "label": item.label, "confidence": item.confidence,
                 "bbox_xyxy": list(item.bbox_xyxy)} for item in result.detections
            ],
        }
    encoded = json.dumps(data, allow_nan=False).encode("utf-8")
    if len(encoded) > _RESULT_BYTES:
        raise DetectionError("Detection result exceeds the bounded worker transport size.")
    return encoded


def _worker_main(connection, shared_name: str, frame_bytes: int, camera, models, detection) -> None:
    session = None
    memory = None
    try:
        memory = shared_memory.SharedMemory(name=shared_name)
        session = _PiSession(camera, models, detection)
        connection.send(("ready", 0))
        frame_id = 0
        while True:
            command = connection.recv()
            if command == "close":
                break
            if command != "read":
                raise CaptureError("Invalid camera worker command.")
            frame_id += 1
            packet = session.read_packet(frame_id)
            if packet is None:
                connection.send(("end", 0))
                continue
            if packet.image_bgr.nbytes > frame_bytes:
                raise CaptureError("Camera output exceeds the configured shared frame buffer.")
            encoded = _packet_metadata(packet)
            image_view = np.ndarray(packet.image_bgr.shape, dtype=np.uint8, buffer=memory.buf)
            image_view[:] = packet.image_bgr
            del image_view
            memory.buf[frame_bytes:frame_bytes + len(encoded)] = encoded
            # A small control message follows all bulk writes. No frame or large
            # pickle is read from a pipe after its poll deadline has elapsed.
            connection.send(("frame", len(encoded)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        kind = "detection_error" if isinstance(exc, DetectionError) else "capture_error"
        try:
            connection.send((kind, str(exc)[:1024]))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            if memory is not None:
                memory.close()
            connection.close()


class Imx500Capture:
    """Own a spawned Pi camera worker with finite startup/read/shutdown waits."""

    def __init__(self, camera_settings: CameraSettings, model_settings: ModelSettings,
                 detection_settings: DetectionSettings) -> None:
        self.camera_settings = camera_settings
        self.model_settings = model_settings
        self.detection_settings = detection_settings
        self._connection = None
        self._process = None
        self._memory = None
        self._closed = False
        self._opened = False
        self._last_frame_id = 0
        self._frame_bytes = (camera_settings.width + 32) * (camera_settings.height + 2) * 3

    def __enter__(self) -> Imx500Capture:
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _open(self) -> None:
        if self._closed:
            raise CaptureError("The AI Camera source is closed. Create a new Imx500Capture.")
        if self._opened:
            return
        if sys.platform != "linux":
            raise CaptureError("IMX500 capture requires Raspberry Pi OS/Linux; use --source sample on macOS.")
        backend = self.model_settings.object_detection_backend
        if backend not in (None, "none", "imx500"):
            raise DetectionError("The AI Camera source supports the imx500 or none detector backend.")
        if backend == "imx500":
            path = self.model_settings.object_detection_path
            if path is None or not path.is_file():
                raise DetectionError(f"Local IMX500 model file does not exist: {path}")
            if path.name != SUPPORTED_MODEL:
                raise DetectionError(f"This decoder supports {SUPPORTED_MODEL} only; other models need their own decoder.")
        child = None
        try:
            context = multiprocessing.get_context("spawn")
            self._memory = shared_memory.SharedMemory(create=True, size=self._frame_bytes + _RESULT_BYTES)
            self._connection, child = context.Pipe()
            self._process = context.Process(
                target=_worker_main,
                args=(child, self._memory.name, self._frame_bytes, self.camera_settings,
                      self.model_settings, self.detection_settings), daemon=True,
            )
            logger.info("Starting AI Camera; local model firmware transfer may take several minutes.")
            self._process.start()
            child.close()
            child = None
            kind, _ = self._receive(self.camera_settings.startup_timeout_s, "startup/model transfer")
            if kind != "ready":
                raise CaptureError("Camera worker did not complete startup.")
            self._opened = True
            logger.info("AI Camera worker ready; missing inference outputs will be reported as not_ready.")
        except BaseException:
            if child is not None:
                child.close()
            self.close()
            raise

    def _receive(self, timeout_s: float, stage: str):
        try:
            if not self._connection.poll(timeout_s):
                raise CaptureError(f"AI Camera {stage} timed out after {timeout_s:g} seconds.")
            message = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise CaptureError(f"AI Camera worker stopped during {stage}.") from exc
        if not isinstance(message, tuple) or len(message) != 2:
            raise CaptureError("Invalid camera worker response.")
        kind, value = message
        if kind == "detection_error":
            raise DetectionError(f"IMX500 detector: {value}")
        if kind == "capture_error":
            raise CaptureError(f"AI Camera: {value}")
        return kind, value

    def read_packet(self) -> FramePacket | None:
        self._open()
        try:
            self._connection.send("read")
            kind, length = self._receive(self.camera_settings.capture_timeout_s, "capture")
            if kind == "end":
                return None
            if kind != "frame" or type(length) is not int or not 0 < length <= _RESULT_BYTES:
                raise CaptureError("Invalid frame response from the AI Camera worker.")
            raw = bytes(self._memory.buf[self._frame_bytes:self._frame_bytes + length])
            data = json.loads(raw.decode("utf-8"))
            shape = data["shape"]
            if (
                not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
                or any(type(value) is not int or value <= 0 for value in shape)
                or math.prod(shape) > self._frame_bytes
                or type(data["frame_id"]) is not int or data["frame_id"] <= self._last_frame_id
            ):
                raise CaptureError("Invalid or stale AI Camera frame metadata.")
            image = np.ndarray(tuple(shape), dtype=np.uint8, buffer=self._memory.buf).copy()
            inference = data["inference"]
            result = None
            if inference is not None:
                if (
                    inference["backend"] != "imx500" or inference["status"] not in ("ready", "not_ready")
                    or inference["simulated"] is not False
                    or (inference["status"] == "not_ready" and inference["detections"])
                ):
                    raise CaptureError("Invalid AI Camera inference status.")
                detections = tuple(
                    Detection(item["class_id"], item["label"], item["confidence"], tuple(item["bbox_xyxy"]))
                    for item in inference["detections"]
                )
                result = DetectionResult(data["frame_id"], inference["backend"], inference["status"],
                                         detections, inference["simulated"])
            self._last_frame_id = data["frame_id"]
            return FramePacket(data["frame_id"], data["captured_monotonic_ns"], image, result)
        except (CaptureError, DetectionError):
            self.close()
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.close()
            raise CaptureError(f"Cannot read AI Camera worker result: {exc}") from exc

    def close(self) -> None:
        """Try graceful shutdown, then terminate/kill a blocked hardware worker."""
        if self._closed:
            return
        self._closed = True
        self._opened = False
        connection, self._connection = self._connection, None
        process, self._process = self._process, None
        memory, self._memory = self._memory, None
        try:
            if connection is not None:
                try:
                    connection.send("close")
                except (BrokenPipeError, EOFError, OSError):
                    pass
                connection.close()
            if process is not None and process.pid is not None:
                process.join(_JOIN_TIMEOUT_S)
                if process.is_alive():
                    process.terminate()
                    process.join(_JOIN_TIMEOUT_S)
                if process.is_alive():
                    process.kill()
                    process.join(_JOIN_TIMEOUT_S)
                if process.is_alive():
                    logger.error("AI Camera worker did not exit after termination; verify the camera before retrying.")
                else:
                    process.close()
        finally:
            if memory is not None:
                try:
                    memory.close()
                finally:
                    try:
                        memory.unlink()
                    except FileNotFoundError:
                        pass
