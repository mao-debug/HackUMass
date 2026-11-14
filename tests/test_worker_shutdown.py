"""Worker signal ownership and cleanup without cameras or real shared memory."""

import signal
import threading
from unittest.mock import Mock

import pytest

from vidi.camera import managed


def fake_resources(monkeypatch):
    connection, memory, session = Mock(), Mock(), Mock()
    connection.recv.return_value = "close"
    memory_factory = Mock(return_value=memory)
    session_factory = Mock(return_value=session)
    monkeypatch.setattr(managed.shared_memory, "SharedMemory", memory_factory)
    return connection, memory, session, memory_factory, session_factory


def test_spawned_worker_ignores_sigint_before_opening_resources_and_leaves_sigterm_untouched(monkeypatch):
    connection, memory, session, memory_factory, session_factory = fake_resources(monkeypatch)
    monkeypatch.setattr(managed.multiprocessing, "parent_process", lambda: object())
    events = []
    policy = Mock(side_effect=lambda signum, handler: events.append(("signal", signum, handler)))
    monkeypatch.setattr(managed.signal, "signal", policy)
    memory_factory.side_effect = lambda **kwargs: events.append(("memory", kwargs)) or memory
    session_factory.side_effect = lambda: events.append(("session",)) or session
    managed.run_worker(connection, "fake-memory", 100, session_factory, ())
    assert events[:3] == [
        ("signal", signal.SIGINT, signal.SIG_IGN), ("memory", {"name": "fake-memory"}), ("session",),
    ]
    policy.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)
    session.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    connection.close.assert_called_once_with()


@pytest.mark.parametrize("stage", ["startup", "capture", "receive"])
def test_keyboard_interrupt_is_quiet_and_releases_every_open_resource(monkeypatch, stage):
    connection, memory, session, _, session_factory = fake_resources(monkeypatch)
    policy = Mock()
    monkeypatch.setattr(managed.signal, "signal", policy)
    monkeypatch.setattr(managed.multiprocessing, "parent_process", lambda: object())
    if stage == "startup":
        session_factory.side_effect = KeyboardInterrupt
    elif stage == "capture":
        connection.recv.return_value = "read"
        session.read_packet.side_effect = KeyboardInterrupt
    else:
        connection.recv.side_effect = KeyboardInterrupt
    managed.run_worker(connection, "fake-memory", 100, session_factory, ())
    if stage == "startup":
        session.close.assert_not_called()  # No session was returned to the worker.
        connection.send.assert_not_called()
    else:
        session.close.assert_called_once_with()
        connection.send.assert_called_once_with(("ready", 0))
    memory.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_cleanup_interrupt_or_failure_does_not_skip_remaining_resources(monkeypatch, caplog):
    connection, memory, session, _, session_factory = fake_resources(monkeypatch)
    monkeypatch.setattr(managed.multiprocessing, "parent_process", lambda: None)
    session.close.side_effect = KeyboardInterrupt
    memory.close.side_effect = RuntimeError("memory close failed")
    managed.run_worker(connection, "fake-memory", 100, session_factory, ())
    session.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    connection.close.assert_called_once_with()
    assert "could not close shared memory" in caplog.text


def test_direct_parent_caller_does_not_change_parent_signal_handler(monkeypatch):
    connection, _, _, _, session_factory = fake_resources(monkeypatch)
    monkeypatch.setattr(managed.multiprocessing, "parent_process", lambda: None)
    policy = Mock()
    monkeypatch.setattr(managed.signal, "signal", policy)
    managed.run_worker(connection, "fake-memory", 100, session_factory, ())
    policy.assert_not_called()


def test_threaded_caller_does_not_attempt_illegal_signal_handler_change(monkeypatch):
    connection, memory, session, _, session_factory = fake_resources(monkeypatch)
    monkeypatch.setattr(managed.multiprocessing, "parent_process", lambda: object())
    policy = Mock()
    monkeypatch.setattr(managed.signal, "signal", policy)
    errors = []

    def run():
        try:
            managed.run_worker(connection, "fake-memory", 100, session_factory, ())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert errors == []
    policy.assert_not_called()
    session.close.assert_called_once_with()
    memory.close.assert_called_once_with()
    connection.close.assert_called_once_with()
