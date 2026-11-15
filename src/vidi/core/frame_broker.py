"""Single camera owner publishing only the latest frame, independent of work."""

from __future__ import annotations

import threading
import time

from vidi.core.types import FramePacket


class FrameUnavailable(RuntimeError):
    """No recent frame is available; old pixels must not masquerade as current."""


class FrameBroker:
    def __init__(self, settings, *, source_factory=None):
        if source_factory is None:
            from vidi.core.controller import _open_source
            source_factory = _open_source
        self.settings = settings
        self._factory = source_factory
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._epoch = 0
        self._paused = False
        self._latest = None
        self._source = None
        self._thread = None
        self.error = None
        self.frames = 0
        self.first_ns = self.last_ns = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="vidi-camera-broker")
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            with self._condition:
                if self._paused:
                    self._condition.wait(.05)
                    continue
            source = None
            try:
                self._restart.clear()
                epoch = self._epoch
                source = self._factory(self.settings)
                with self._condition:
                    if self._paused or self._stop.is_set() or epoch != self._epoch:
                        continue
                    self._source = source
                with source:
                    while not self._stop.is_set():
                        with self._condition:
                            if self._paused or epoch != self._epoch:
                                break
                        started = time.monotonic()
                        if hasattr(source, "read_packet"):
                            packet = source.read_packet()
                        else:
                            image = source.read()
                            packet = None if image is None else FramePacket(0, time.monotonic_ns(), image)
                        if packet is None:
                            raise FrameUnavailable("Camera did not return a frame.")
                        with self._condition:
                            if self._paused or self._stop.is_set() or epoch != self._epoch:
                                break
                            self.frames += 1
                            self.first_ns = self.first_ns or packet.captured_monotonic_ns
                            self.last_ns = packet.captured_monotonic_ns
                            self._latest = FramePacket(self.frames, packet.captured_monotonic_ns,
                                                       packet.image_bgr.copy())
                            self.error = None
                            self._condition.notify_all()
                        self._stop.wait(max(0., 1. / self.settings.camera.fps - (time.monotonic() - started)))
            except Exception:
                with self._condition:
                    self._latest = None
                    if not self._paused and not self._stop.is_set():
                        self.error = "Camera unavailable. Check its connection and retry."
                    self._condition.notify_all()
                # Device failures require explicit retry; never busy-loop reopen.
                with self._condition:
                    while not self._paused and not self._stop.is_set() and not self._restart.is_set():
                        self._condition.wait(.1)
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception:
                        pass
                with self._condition:
                    self._source = None

    def latest(self, *, max_age_s=2., after_ns=None):
        with self._condition:
            packet = self._latest
            now = time.monotonic_ns()
            if (packet is None or self._paused or not 0 <= now - packet.captured_monotonic_ns <= max_age_s * 1e9
                    or (after_ns is not None and packet.captured_monotonic_ns <= after_ns)):
                raise FrameUnavailable("A fresh camera frame is unavailable.")
            return FramePacket(packet.frame_id, packet.captured_monotonic_ns, packet.image_bgr.copy())

    def wait_fresh(self, *, after_ns, cancel_event, timeout_s=3., max_age_s=2.):
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not cancel_event.is_set():
                try:
                    return self.latest(max_age_s=max_age_s, after_ns=after_ns)
                except FrameUnavailable:
                    if self._paused or self._stop.is_set() or time.monotonic() >= deadline:
                        break
                    self._condition.wait(.03)
        raise FrameUnavailable("A new camera frame was not available in time.")

    def pause(self):
        with self._condition:
            self._paused = True
            self._epoch += 1
            self._latest = None
            source = self._source
            self._condition.notify_all()
        if source is not None:
            try:
                source.close()
            except Exception:
                pass

    def resume(self):
        with self._condition:
            self._paused = False
            self._restart.set()
            self.error = None
            self._condition.notify_all()

    def close(self):
        self._stop.set()
        self.pause()
        if self._thread is not None:
            self._thread.join(timeout=1.)

