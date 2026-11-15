"""A blocking capture must not block callers or republish after privacy pause."""

from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from vidi.core.configuration import AppSettings
from vidi.core.frame_broker import FrameBroker, FrameUnavailable


class Source:
    def __init__(self):
        self.closed = False
        self.entered = False

    def __enter__(self):
        assert not self.closed
        self.entered = True
        return self

    def __exit__(self, *args):
        self.close()

    def read(self):
        return np.full((10, 10, 3), 42, np.uint8)

    def close(self):
        self.closed = True


def settings():
    original = AppSettings()
    return replace(original, camera=replace(original.camera, fps=100))


def test_latest_owned_frames_no_backlog_and_pause_resume():
    sources = []

    def factory(_):
        source = Source()
        sources.append(source)
        return source

    broker = FrameBroker(settings(), source_factory=factory).start()
    cancel = threading.Event()
    try:
        first = broker.wait_fresh(after_ns=time.monotonic_ns() - 1_000_000_000, cancel_event=cancel)
        first.image_bgr[:] = 0
        assert broker.latest().image_bgr[0, 0, 0] == 42
        broker.pause()
        with pytest.raises(FrameUnavailable):
            broker.latest()
        assert sources[0].closed
        broker.resume()
        next_frame = broker.wait_fresh(after_ns=first.captured_monotonic_ns, cancel_event=cancel)
        assert next_frame.frame_id > first.frame_id
        assert len(sources) >= 2
    finally:
        broker.close()


def test_pause_while_source_is_being_created_does_not_open_it():
    creating, release = threading.Event(), threading.Event()
    source = Source()

    def factory(_):
        creating.set()
        release.wait(1)
        return source

    broker = FrameBroker(settings(), source_factory=factory).start()
    try:
        assert creating.wait(1)
        broker.pause()
        release.set()
        deadline = time.monotonic() + 1
        while not source.closed and time.monotonic() < deadline:
            time.sleep(.001)
        assert source.closed and not source.entered
        with pytest.raises(FrameUnavailable):
            broker.latest()
    finally:
        broker.close()


def test_blocked_read_cannot_publish_cancelled_frame():
    entered, release = threading.Event(), threading.Event()

    class Blocked(Source):
        def read(self):
            entered.set()
            release.wait(1)
            return super().read()

    source = Blocked()
    broker = FrameBroker(settings(), source_factory=lambda _: source).start()
    try:
        assert entered.wait(1)
        broker.pause()
        release.set()
        time.sleep(.02)
        assert broker.frames == 0
    finally:
        broker.close()


def test_missing_camera_stays_unavailable_until_explicit_retry():
    calls = []

    def factory(_):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("private driver details")
        return Source()

    broker = FrameBroker(settings(), source_factory=factory).start()
    try:
        deadline = time.monotonic() + 1
        while broker.error is None and time.monotonic() < deadline:
            time.sleep(.001)
        assert broker.error and "private" not in broker.error
        time.sleep(.02)
        assert len(calls) == 1
        broker.pause()
        broker.resume()
        assert broker.wait_fresh(after_ns=0, cancel_event=threading.Event())
    finally:
        broker.close()

