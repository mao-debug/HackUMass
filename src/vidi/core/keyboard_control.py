"""Read only explicit recovery keys from a selected Linux input device."""

import os
from pathlib import Path
import select
import struct
import threading


class KeyboardControl:
    """No grabs, key logging, root execution, or automatic permission changes."""

    def __init__(self, path, callback):
        self.path = Path(path)
        self.callback = callback
        self._stop = threading.Event()
        self._fd = None
        self._thread = None
        self.error = None

    def start(self):
        # Restrict this interface to kernel event devices, not arbitrary files.
        resolved = self.path.resolve()
        if resolved.parent != Path("/dev/input") or not resolved.name.startswith("event"):
            raise ValueError("Choose a keyboard /dev/input/event device or by-id event-kbd symlink.")
        self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self._thread = threading.Thread(target=self._run, daemon=True, name="vidi-recovery-keys")
        try:
            self._thread.start()
        except Exception:
            os.close(self._fd)
            self._fd = self._thread = None
            raise OSError("Recovery keyboard listener could not start.") from None
        return self

    def _run(self):
        event = struct.Struct("@llHHi")
        data = b""
        try:
            while not self._stop.is_set():
                if not select.select([self._fd], [], [], .1)[0]:
                    continue
                chunk = os.read(self._fd, event.size * 16)
                if not chunk:
                    raise OSError
                data += chunk
                while len(data) >= event.size:
                    _, _, kind, code, value = event.unpack(data[:event.size])
                    data = data[event.size:]
                    if kind == 1 and value == 1:
                        action = {57: "stop", 88: "privacy_toggle", 1: "quit",
                                  19: "retry", 35: "help", 48: "scan", 46: "confirm",
                                  45: "reject", 50: "wake", 32: "describe", 20: "read_text",
                                  25: "proximity", 2: "mode_normal", 3: "mode_chess"}.get(code)
                        if action:
                            self.callback(action)
        except (OSError, ValueError, TypeError):
            if not self._stop.is_set():
                self.error = "Recovery keyboard disconnected. Assistant input is paused."

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=.3)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
