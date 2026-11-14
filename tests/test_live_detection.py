"""Latest-frame inference scheduling with fake detectors and controlled events."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vidi.ai import live_detection as module
from vidi.ai.live_detection import LiveDetector
from vidi.ai.object_detection import DetectionError
from vidi.core.types import DetectionResult, FramePacket


def packet(frame_id, value=10):
    return FramePacket(frame_id, frame_id * 100, np.full((4, 6, 3), value, np.uint8))


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    pause = threading.Event()
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        pause.wait(0.001)
    pytest.fail("Inference worker did not reach the expected state")


def result_for(worker, frame_id):
    def matching_pair():
        pair = worker.poll()
        return pair if pair is not None and pair[0].frame_id == frame_id else None

    return wait_until(matching_pair)


def cooldown_event(worker, monkeypatch):
    entered = threading.Event()
    original_wait = worker._condition.wait

    def record_wait(timeout=None):
        if timeout is not None:
            entered.set()
        return original_wait(timeout=timeout)

    monkeypatch.setattr(worker._condition, "wait", record_wait)
    return entered


def test_submit_does_not_wait_for_inference_and_pending_slot_keeps_only_latest_owned_frame():
    started, release = threading.Event(), threading.Event()
    calls, seen, threads = [], [], []

    def detect(frame):
        calls.append(frame.frame_id)
        threads.append(threading.get_ident())
        if frame.frame_id == 1:
            started.set()
            assert release.wait(2)
        seen.append((frame.frame_id, frame.image_bgr.copy()))
        return DetectionResult(frame.frame_id, "fixture", "ready", simulated=True)

    worker = LiveDetector(SimpleNamespace(detect=detect))
    first, newest = packet(1, 10), packet(3, 30)
    try:
        with worker:
            worker.submit(first)
            assert started.wait(1)
            first.image_bgr[:] = 0
            submitted = time.monotonic()
            worker.submit(packet(2, 20))
            worker.submit(newest)
            assert time.monotonic() - submitted < 0.5
            newest.image_bgr[:] = 0
            assert calls == [1] and worker.busy is True
            release.set()
            owned, result = result_for(worker, 3)
            assert owned.frame_id == result.frame_id == 3
            np.testing.assert_array_equal(owned.image_bgr, np.full((4, 6, 3), 30, np.uint8))
            assert calls == [1, 3]
            assert [int(image[0, 0, 0]) for _, image in seen] == [10, 30]
            assert len(set(threads)) == 1 and threads[0] != threading.get_ident()
            assert worker.poll() is None
    finally:
        release.set()
        worker.close()


def test_completed_output_slot_replaces_unpolled_results_and_consumes_once():
    detector = Mock()
    detector.detect.side_effect = lambda frame: DetectionResult(frame.frame_id, "fixture", "ready")
    with LiveDetector(detector) as worker:
        for frame_id in (1, 2, 3):
            worker.submit(packet(frame_id))
            wait_until(lambda: not worker.busy)
        owned, result = worker.poll()
        assert owned.frame_id == result.frame_id == 3
        assert worker.poll() is None and detector.detect.call_count == 3


def test_close_drops_pending_work_and_bounds_wait_for_blocked_daemon(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []

    def detect(frame):
        calls.append(frame.frame_id)
        started.set()
        assert release.wait(2)
        return DetectionResult(frame.frame_id, "fixture", "ready")

    monkeypatch.setattr(module, "_JOIN_TIMEOUT_S", 0.03)
    worker = LiveDetector(SimpleNamespace(detect=detect))
    try:
        worker.submit(packet(1))
        assert started.wait(1)
        worker.submit(packet(2))
        began = time.monotonic()
        worker.close()
        assert time.monotonic() - began < 0.5
        assert worker._thread.daemon and worker.stopped and not worker.busy
        assert worker.poll() is None
        with pytest.raises(DetectionError, match="closed"):
            worker.submit(packet(3))
        release.set()
        worker._thread.join(timeout=1)
        assert not worker._thread.is_alive() and calls == [1]
        worker.close()
    finally:
        release.set()
        worker.close()


@pytest.mark.parametrize("mismatch", [False, True])
def test_worker_failure_or_mismatched_frame_surfaces_safely_and_stops(mismatch, capsys, caplog):
    detector = Mock()
    if mismatch:
        detector.detect.return_value = DetectionResult(99, "fixture", "ready")
    else:
        detector.detect.side_effect = RuntimeError("PRIVATE engine diagnostic")
    with LiveDetector(detector) as worker:
        worker.submit(packet(1))

        def failure():
            try:
                worker.poll()
            except DetectionError as error:
                return error
            return None

        error = wait_until(failure)
        assert "PRIVATE" not in str(error)
        assert ("different frame" if mismatch else "Live detection failed") in str(error)
        assert worker.stopped and not worker.busy
        worker.submit(packet(2))
        detector.detect.assert_called_once()
    assert capsys.readouterr() == ("", "") and caplog.text == ""


def test_cooldown_starts_after_completion_and_waiting_input_stays_latest_only(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    started, release = threading.Event(), threading.Event()
    calls = []

    def detect(frame):
        calls.append((frame.frame_id, now[0]))
        if frame.frame_id == 1:
            started.set()
            assert release.wait(2)
            now[0] = 2.0  # Model work took two seconds in the fake clock.
        return DetectionResult(frame.frame_id, "gemini", "ready")

    worker = LiveDetector(SimpleNamespace(detect=detect), min_interval_s=5)
    waiting = cooldown_event(worker, monkeypatch)
    try:
        with worker:
            worker.submit(packet(1))
            assert started.wait(1)
            release.set()
            result_for(worker, 1)
            worker.submit(packet(2))
            assert waiting.wait(1)
            waiting.clear()
            now[0] = 6.0
            worker.submit(packet(3))
            assert waiting.wait(1)
            assert calls == [(1, 0.0)]  # Five seconds from start was insufficient.
            now[0] = 7.0
            worker.submit(packet(4))
            result_for(worker, 4)
            assert calls == [(1, 0.0), (4, 7.0)]
    finally:
        release.set()
        worker.close()


def test_close_wakes_long_cooldown_without_waiting_for_interval(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    detector = Mock()
    detector.detect.side_effect = lambda frame: DetectionResult(frame.frame_id, "gemini", "ready")
    worker = LiveDetector(detector, min_interval_s=5)
    waiting = cooldown_event(worker, monkeypatch)
    try:
        worker.submit(packet(1))
        result_for(worker, 1)
        worker.submit(packet(2))
        assert waiting.wait(1)
        started = time.monotonic()
        worker.close()
        assert time.monotonic() - started < 0.5
        assert not worker._thread.is_alive()
        detector.detect.assert_called_once()
    finally:
        worker.close()


def test_nonready_result_is_published_once_then_stops_and_discards_waiting_frame():
    started, release = threading.Event(), threading.Event()
    calls = []

    def detect(frame):
        calls.append(frame.frame_id)
        started.set()
        assert release.wait(2)
        return DetectionResult(frame.frame_id, "gemini", "unavailable", error="API quota unavailable")

    worker = LiveDetector(SimpleNamespace(detect=detect), stop_on_unavailable=True)
    try:
        with worker:
            worker.submit(packet(1))
            assert started.wait(1)
            worker.submit(packet(2))
            release.set()
            owned, result = result_for(worker, 1)
            assert owned.frame_id == result.frame_id == 1 and result.status == "unavailable"
            assert result.error == "API quota unavailable" and worker.stopped
            assert worker.poll() is None
            worker.submit(packet(3))
            assert calls == [1]
    finally:
        release.set()
        worker.close()


def test_idle_context_closes_without_detecting_or_restarting():
    detector = Mock()
    worker = LiveDetector(detector)
    assert not worker._thread.is_alive()
    with worker:
        assert worker.busy is False and worker.poll() is None
    assert worker.stopped and not worker._thread.is_alive()
    detector.detect.assert_not_called()
    with pytest.raises(DetectionError, match="closed"):
        worker.__enter__()
