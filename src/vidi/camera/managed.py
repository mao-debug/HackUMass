"""Bounded process and shared-memory transport shared by Pi camera adapters.

This module imports no Raspberry Pi camera packages. Hardware sessions are
constructed by adapter-specific worker entry points after process startup.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
from multiprocessing import shared_memory
import sys
from typing import TYPE_CHECKING

import numpy as np

from vidi.ai.object_detection import DetectionError
from vidi.camera.capture import CaptureError
from vidi.core.types import Detection, DetectionResult, FramePacket

if TYPE_CHECKING:
    from vidi.core.configuration import CameraSettings

logger = logging.getLogger(__name__)
_RESULT_BYTES = 65536
_JOIN_TIMEOUT_S = 1.0


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


def run_worker(connection, shared_name: str, frame_bytes: int, session_factory, session_arguments: tuple) -> None:
    session = None
    memory = None
    try:
        memory = shared_memory.SharedMemory(name=shared_name)
        session = session_factory(*session_arguments)
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


class ManagedPiCapture:
    """Own a spawned Pi camera worker with finite startup/read/shutdown waits."""

    def __init__(self, camera_settings: CameraSettings, worker_target, worker_arguments: tuple,
                 *, source_name: str, startup_stage: str = "startup",
                 startup_message: str | None = None, ready_message: str | None = None,
                 accept_inference: bool = False) -> None:
        self.camera_settings = camera_settings
        self._worker_target = worker_target
        self._worker_arguments = worker_arguments
        self._source_name = source_name
        self._startup_stage = startup_stage
        self._startup_message = startup_message or f"Starting {source_name} camera-only capture; inference runs separately."
        self._ready_message = ready_message or f"{source_name} worker ready for camera-only capture; inference runs separately."
        self._accept_inference = accept_inference
        self._connection = None
        self._process = None
        self._memory = None
        self._closed = False
        self._opened = False
        self._last_frame_id = 0
        self._frame_bytes = (camera_settings.width + 32) * (camera_settings.height + 2) * 3

    def __enter__(self) -> ManagedPiCapture:
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _validate_source(self) -> None:
        """Subclasses validate their local assets before any process is started."""

    def _open(self) -> None:
        if self._closed:
            raise CaptureError(f"The {self._source_name} source is closed. Create a new {type(self).__name__}.")
        if self._opened:
            return
        if sys.platform != "linux":
            raise CaptureError(f"{self._source_name} capture requires Raspberry Pi OS/Linux; use --source sample on macOS.")
        self._validate_source()
        child = None
        try:
            context = multiprocessing.get_context("spawn")
            self._memory = shared_memory.SharedMemory(create=True, size=self._frame_bytes + _RESULT_BYTES)
            self._connection, child = context.Pipe()
            self._process = context.Process(
                target=self._worker_target,
                args=(child, self._memory.name, self._frame_bytes, *self._worker_arguments), daemon=True,
            )
            logger.info("%s", self._startup_message)
            self._process.start()
            child.close()
            child = None
            kind, _ = self._receive(self.camera_settings.startup_timeout_s, self._startup_stage)
            if kind != "ready":
                raise CaptureError("Camera worker did not complete startup.")
            self._opened = True
            logger.info("%s", self._ready_message)
        except BaseException:
            if child is not None:
                child.close()
            self.close()
            raise

    def _receive(self, timeout_s: float, stage: str):
        try:
            if not self._connection.poll(timeout_s):
                raise CaptureError(f"{self._source_name} {stage} timed out after {timeout_s:g} seconds.")
            message = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise CaptureError(f"{self._source_name} worker stopped during {stage}.") from exc
        if not isinstance(message, tuple) or len(message) != 2:
            raise CaptureError("Invalid camera worker response.")
        kind, value = message
        if kind == "detection_error":
            raise DetectionError(f"{self._source_name} detector: {value}")
        if kind == "capture_error":
            raise CaptureError(f"{self._source_name}: {value}")
        return kind, value

    def read_packet(self) -> FramePacket | None:
        self._open()
        try:
            self._connection.send("read")
            kind, length = self._receive(self.camera_settings.capture_timeout_s, "capture")
            if kind == "end":
                return None
            if kind != "frame" or type(length) is not int or not 0 < length <= _RESULT_BYTES:
                raise CaptureError(f"Invalid frame response from the {self._source_name} worker.")
            raw = bytes(self._memory.buf[self._frame_bytes:self._frame_bytes + length])
            data = json.loads(raw.decode("utf-8"))
            shape = data["shape"]
            if (
                not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
                or any(type(value) is not int or value <= 0 for value in shape)
                or math.prod(shape) > self._frame_bytes
                or type(data["frame_id"]) is not int or data["frame_id"] <= self._last_frame_id
            ):
                raise CaptureError(f"Invalid or stale {self._source_name} frame metadata.")
            image = np.ndarray(tuple(shape), dtype=np.uint8, buffer=self._memory.buf).copy()
            inference = data["inference"]
            result = None
            if inference is not None:
                if not self._accept_inference:
                    raise CaptureError("Camera-only worker returned unexpected inference metadata.")
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
            raise CaptureError(f"Cannot read {self._source_name} worker result: {exc}") from exc

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
                    logger.error("%s worker did not exit after termination; verify the camera before retrying.", self._source_name)
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
