"""Deterministic process doubles expose privacy/startup/read races, no devices."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from vidi.camera import managed
from vidi.camera.capture import CaptureError
from vidi.core.configuration import CameraSettings


def capture():
    return managed.ManagedPiCapture(CameraSettings(source="picamera2", width=80, height=60),
                                    lambda *_: None, (), source_name="test camera")


def launch(operation):
    errors = []
    completed = threading.Event()
    def run():
        try:
            operation()
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, completed, errors


def transport(monkeypatch):
    monkeypatch.setattr(managed.sys, "platform", "linux")
    memory = SimpleNamespace(name="fake-memory", buf=bytearray(100000), close=Mock(), unlink=Mock())
    factory = Mock(return_value=memory)
    monkeypatch.setattr(managed.shared_memory, "SharedMemory", factory)
    connection = Mock()
    connection.poll.return_value = True
    connection.recv.return_value = ("ready", 0)
    child = Mock()
    process = Mock(pid=123)
    process.is_alive.return_value = False
    context = SimpleNamespace(Pipe=Mock(return_value=(connection, child)), Process=Mock(return_value=process))
    monkeypatch.setattr(managed.multiprocessing, "get_context", lambda *_: context)
    return SimpleNamespace(memory=memory, factory=factory, connection=connection,
                           child=child, process=process, context=context)


def test_close_during_validation_never_allocates_or_starts_worker(monkeypatch):
    fake = transport(monkeypatch)
    source = capture()
    validating, release = threading.Event(), threading.Event()
    def validate():
        validating.set()
        assert release.wait(2)
    monkeypatch.setattr(source, "_validate_source", validate)
    thread, done, errors = launch(source.__enter__)
    assert validating.wait(1)
    source.close()
    release.set()
    assert done.wait(1)
    thread.join(1)
    assert len(errors) == 1 and isinstance(errors[0], CaptureError)
    fake.factory.assert_not_called()
    fake.context.Process.assert_not_called()
    assert source._closed and not source._opened


def test_allocation_and_start_cannot_outlive_atomic_close(monkeypatch):
    fake = transport(monkeypatch)
    source = capture()
    allocating, release, closing = threading.Event(), threading.Event(), threading.Event()
    def allocate(**kwargs):
        allocating.set()
        assert release.wait(2)
        return fake.memory
    fake.factory.side_effect = allocate
    opening_thread, opening_done, opening_errors = launch(source.__enter__)
    assert allocating.wait(1)
    def close():
        closing.set()
        source.close()
    closing_thread, closing_done, closing_errors = launch(close)
    assert closing.wait(1)
    # The close caller cannot detach partially allocated resources.
    assert not closing_done.is_set()
    release.set()
    assert closing_done.wait(1) and opening_done.wait(1)
    closing_thread.join(1)
    opening_thread.join(1)
    assert not closing_errors
    assert all(isinstance(error, CaptureError) for error in opening_errors)
    fake.process.start.assert_called_once()
    fake.process.close.assert_called_once()
    fake.memory.close.assert_called_once()
    fake.memory.unlink.assert_called_once()
    assert source._closed and not source._opened
    assert source._connection is source._process is source._memory is None


@pytest.mark.parametrize("stage", ["startup", "read"])
def test_close_interrupts_blocked_receive_without_waiting_for_poll_deadline(monkeypatch, stage):
    fake = transport(monkeypatch)
    source = capture()
    polling, detached = threading.Event(), threading.Event()
    fake.connection.close.side_effect = detached.set
    def poll(timeout):
        polling.set()
        assert detached.wait(1), "close was blocked behind receive"
        raise ValueError("handle is closed")
    if stage == "read":
        source._opened = True
        source._connection, source._process, source._memory = fake.connection, fake.process, fake.memory
    fake.connection.poll.side_effect = poll
    operation = source.__enter__ if stage == "startup" else source.read_packet
    thread, done, errors = launch(operation)
    assert polling.wait(1)
    source.close()
    assert done.wait(1)
    thread.join(1)
    assert len(errors) == 1 and isinstance(errors[0], CaptureError)
    assert "stopped during" in str(errors[0])
    assert source._closed and not source._opened
    fake.memory.close.assert_called_once()
    fake.memory.unlink.assert_called_once()


def test_ready_response_after_close_cannot_mark_source_open(monkeypatch):
    fake = transport(monkeypatch)
    source = capture()
    def receive(*args):
        source.close()
        return "ready", 0
    monkeypatch.setattr(source, "_receive", receive)
    with pytest.raises(CaptureError, match="closed during startup"):
        source.__enter__()
    assert source._closed and not source._opened
    fake.memory.unlink.assert_called_once()


def test_detached_receive_and_read_are_controlled_errors(monkeypatch):
    fake = transport(monkeypatch)
    source = capture()
    source.close()
    with pytest.raises(CaptureError, match="stopped during"):
        source._receive(30, "startup")
    with pytest.raises(CaptureError, match="source is closed"):
        source.read_packet()
    fake.factory.assert_not_called()


def test_concurrent_open_cannot_spawn_second_worker(monkeypatch):
    fake = transport(monkeypatch)
    source = capture()
    polling, release = threading.Event(), threading.Event()
    def poll(timeout):
        polling.set()
        return release.wait(2)
    fake.connection.poll.side_effect = poll
    thread, done, errors = launch(source.__enter__)
    assert polling.wait(1)
    with pytest.raises(CaptureError, match="already starting"):
        source.__enter__()
    release.set()
    assert done.wait(1)
    thread.join(1)
    assert not errors
    fake.process.start.assert_called_once()
    source.close()
    fake.memory.unlink.assert_called_once()


def test_simultaneous_close_detaches_resources_once_and_keeps_bounded_escalation(monkeypatch):
    fake = transport(monkeypatch)
    source = capture()
    source._opened = True
    source._connection, source._process, source._memory = fake.connection, fake.process, fake.memory
    joining, release = threading.Event(), threading.Event()
    def join(timeout):
        assert timeout == managed._JOIN_TIMEOUT_S
        joining.set()
        assert release.wait(2)
    fake.process.join.side_effect = join
    fake.process.is_alive.side_effect = [True, True, False]
    thread, done, errors = launch(source.close)
    assert joining.wait(1)
    source.close()  # Idempotent even while another thread is reaping resources.
    release.set()
    assert done.wait(1)
    thread.join(1)
    assert not errors
    assert fake.process.join.call_count == 3
    fake.process.terminate.assert_called_once()
    fake.process.kill.assert_called_once()
    fake.connection.close.assert_called_once()
    fake.memory.close.assert_called_once()
    fake.memory.unlink.assert_called_once()


def test_close_cannot_unmap_memory_during_owned_image_copy(monkeypatch):
    fake = transport(monkeypatch)
    source = capture()
    source._opened = True
    source._connection, source._process, source._memory = fake.connection, fake.process, fake.memory
    metadata = json.dumps({"frame_id": 1, "captured_monotonic_ns": 1,
                           "shape": [60, 80, 3], "inference": None}).encode()
    fake.memory.buf[source._frame_bytes:source._frame_bytes + len(metadata)] = metadata
    fake.connection.recv.return_value = "frame", len(metadata)
    copying, release, closing = threading.Event(), threading.Event(), threading.Event()
    def array(*args, **kwargs):
        copying.set()
        assert release.wait(2)
        assert not fake.memory.close.called
        return np.ndarray(*args, **kwargs)
    monkeypatch.setattr(managed, "np", SimpleNamespace(ndarray=array, uint8=np.uint8))
    reader, read_done, errors = launch(source.read_packet)
    assert copying.wait(1)
    def close():
        closing.set()
        source.close()
    closer, close_done, close_errors = launch(close)
    assert closing.wait(1)
    assert not close_done.is_set() and not fake.memory.close.called
    release.set()
    assert read_done.wait(1) and close_done.wait(1)
    reader.join(1)
    closer.join(1)
    assert not close_errors
    assert all(isinstance(error, CaptureError) for error in errors)
    fake.memory.close.assert_called_once()
    fake.memory.unlink.assert_called_once()
