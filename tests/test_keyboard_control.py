"""Real pipe event parsing, without opening a keyboard or granting permissions."""

import os
import struct
import time

import pytest

from vidi.core.keyboard_control import KeyboardControl


def test_only_explicit_press_actions_are_dispatched_and_disconnect_is_reported(monkeypatch):
    read_fd, write_fd = os.pipe()
    actions = []
    monkeypatch.setattr("vidi.core.keyboard_control.os.open", lambda *args: read_fd)
    keyboard = KeyboardControl("/dev/input/event99", actions.append).start()
    event = struct.Struct("@llHHi")
    try:
        # Arbitrary typing, release, and repeat events never become actions.
        data = b"".join(event.pack(0, 0, 1, code, value) for code, value in (
            (30, 1), (57, 0), (57, 2), (57, 1), (88, 1), (19, 1), (1, 1)))
        os.write(write_fd, data[:7])
        os.write(write_fd, data[7:])
        deadline = time.monotonic() + 1
        while len(actions) < 4 and time.monotonic() < deadline:
            time.sleep(.001)
        assert actions == ["stop", "privacy_toggle", "retry", "quit"]
        os.close(write_fd)
        write_fd = None
        while keyboard.error is None and time.monotonic() < deadline:
            time.sleep(.001)
        assert keyboard.error and "disconnected" in keyboard.error
    finally:
        keyboard.close()
        if write_fd is not None:
            os.close(write_fd)


def test_thread_start_failure_closes_selected_descriptor(monkeypatch):
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr("vidi.core.keyboard_control.os.open", lambda *args: read_fd)
    monkeypatch.setattr("vidi.core.keyboard_control.threading.Thread.start",
                        lambda self: (_ for _ in ()).throw(RuntimeError("private")))
    try:
        with pytest.raises(OSError, match="listener could not start"):
            KeyboardControl("/dev/input/event99", lambda *_: None).start()
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        os.close(write_fd)
