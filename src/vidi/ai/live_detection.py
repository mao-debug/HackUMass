"""A single inference thread with latest-only input and output slots.

The caller explicitly selects the detector and any required cloud opt-in. This
module selects no model or service and opens no camera. A running inference
cannot be forcibly cancelled; its daemon thread cannot delay process exit.
"""

from __future__ import annotations

from dataclasses import replace
import math
import threading
import time
from typing import TYPE_CHECKING

from vidi.ai.object_detection import DetectionError

if TYPE_CHECKING:
    from vidi.core.types import DetectionResult, FramePacket


_JOIN_TIMEOUT_S = 1.0


class LiveDetector:
    """Keep UI capture independent of one exclusively owned detector.

    ``submit`` copies a frame and replaces any waiting input. ``poll`` consumes
    the most recent completed packet/result once. The worker starts on context
    entry or the first submission. Closing drops both slots, prevents further
    submissions, and waits at most one second for an in-flight detection.
    """

    def __init__(self, detector, *, min_interval_s: float = 0.0, stop_on_unavailable: bool = False) -> None:
        try:
            valid_interval = (
                type(min_interval_s) in (int, float)
                and math.isfinite(min_interval_s) and min_interval_s >= 0
            )
        except OverflowError:
            valid_interval = False
        if not valid_interval or type(stop_on_unavailable) is not bool:
            raise DetectionError("Live inference needs a finite nonnegative interval and a boolean stop setting.")
        self._detector = detector
        self._min_interval_s = min_interval_s
        self._stop_on_unavailable = stop_on_unavailable
        self._next_start = 0.0
        self._condition = threading.Condition()
        self._pending: FramePacket | None = None
        self._completed: tuple[FramePacket, DetectionResult] | None = None
        self._error: str | None = None
        self._closed = False
        self._stopped = False
        self._started = False
        self._detecting = False
        self._thread = threading.Thread(target=self._run, name="vidi-inference", daemon=True)

    def __enter__(self) -> LiveDetector:
        with self._condition:
            self._start_locked()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _start_locked(self) -> None:
        if self._closed:
            raise DetectionError("The live detection worker is closed.")
        if not self._started:
            try:
                self._thread.start()
            except RuntimeError:
                raise DetectionError("The inference worker could not be started.") from None
            self._started = True

    @property
    def busy(self) -> bool:
        """Whether an inference is running or a newer frame is waiting."""
        with self._condition:
            return not self._closed and (self._detecting or self._pending is not None)

    @property
    def stopped(self) -> bool:
        """Whether closed, failed, or stopped after publishing an unavailable result."""
        with self._condition:
            return self._closed or self._stopped

    def submit(self, packet: FramePacket) -> None:
        """Replace the pending frame without waiting for active inference."""
        with self._condition:
            self._start_locked()
            if self._stopped:
                return  # poll reports the worker's failure without losing it.
        try:
            owned = replace(packet, image_bgr=packet.image_bgr.copy())
        except Exception:
            raise DetectionError("The current frame could not be copied for inference.") from None
        with self._condition:
            if self._closed:
                raise DetectionError("The live detection worker is closed.")
            if self._stopped:
                return
            self._pending = owned
            self._condition.notify()

    def poll(self) -> tuple[FramePacket, DetectionResult] | None:
        """Consume the newest completed pair, or report a sanitized failure."""
        with self._condition:
            if self._closed:
                return None
            if self._error is not None:
                raise DetectionError(self._error)
            completed, self._completed = self._completed, None
            return completed

    def _run(self) -> None:
        while True:
            with self._condition:
                while True:
                    if self._closed:
                        return
                    if self._pending is None:
                        self._condition.wait()
                        continue
                    delay = self._next_start - time.monotonic()
                    if delay > 0:
                        self._condition.wait(timeout=delay)
                        continue
                    break
                packet, self._pending = self._pending, None
                # Claim the in-flight operation before releasing the lock;
                # no subsequent operation can be claimed after close().
                self._detecting = True
            failure = None
            try:
                inference = self._detector.detect(packet)
                if inference.frame_id != packet.frame_id:
                    failure = "Inference returned results for a different frame."
            except BaseException:
                # An uncaught thread exception would print raw diagnostics and
                # leave the UI unaware that inference stopped. Main-thread
                # Ctrl+C is handled by the caller and context cleanup instead.
                failure = "Live detection failed. Check the selected detector and camera input."
            completed_at = time.monotonic()
            with self._condition:
                self._detecting = False
                if self._closed:
                    return
                if failure is not None:
                    self._error = failure
                    self._stopped = True
                    self._pending = self._completed = None
                    return
                self._completed = (packet, inference)
                self._next_start = completed_at + self._min_interval_s
                if self._stop_on_unavailable and inference.status != "ready":
                    self._pending = None
                    self._stopped = True
                    return

    def close(self) -> None:
        """Drop waiting work and results; bound the wait for an active call."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending = self._completed = None
            self._condition.notify_all()
            started = self._started
        if started and self._thread is not threading.current_thread():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
