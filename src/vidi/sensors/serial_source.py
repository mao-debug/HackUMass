"""Optional USB serial protocol v1; acquisition is never inferred from receipt.

Only a matching random request token can produce observations. Every response
must describe NEW measurements acquired after R1, with no legacy-stream fallback.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import threading
import time
from typing import TYPE_CHECKING, Callable

from .fixture import validate_reading
from .proximity import ProximityError, ProximityObservation, ProximitySnapshot

if TYPE_CHECKING:
    from vidi.core.configuration import ProximitySettings

MAX_LINE_BYTES = 512
_MAX_EXCHANGE_BYTES = 8192
_HEX8 = re.compile(r"[0-9a-f]{8}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")


class _ResponseTimeout(ProximityError):
    """No matching response arrived; only HELLO may retry this condition."""


def _unique_pairs(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _no_constant(_value: str) -> None:
    raise ValueError("Nonfinite JSON number")


class SerialSource:
    """One source/one caller; an optional factory permits device-free tests.

    Construction does not import pyserial or open the port. The lexical port
    string is preserved, including stable /dev/serial/by-id symlinks.
    """

    def __init__(self, settings: ProximitySettings, *, serial_factory: Callable | None = None,
                 token_factory: Callable[[], str] | None = None) -> None:
        self.settings = settings
        self._serial_factory = serial_factory
        self._token_factory = token_factory or (lambda: secrets.token_hex(8))
        self._port = None
        self._port_lock = threading.Lock()
        self._closed = threading.Event()
        self._boot: str | None = None
        self._last_seq: int | None = None
        self._last_start_ns: int | None = None

    def _open(self) -> None:
        if self._closed.is_set():
            raise ProximityError("The proximity source is closed.")
        if self._port is not None:
            return
        if not self.settings.port:
            raise ProximityError("An explicit proximity serial port is required.")
        try:
            factory = self._serial_factory
            if factory is None:
                from serial import Serial
                factory = Serial
            # Set control lines before opening to avoid intentional reset/flow
            # control assertions. Some carrier drivers may still reset on open.
            port = factory(port=None, baudrate=self.settings.baud_rate, timeout=.02,
                           write_timeout=min(.1, self.settings.response_timeout_s),
                           rtscts=False, dsrdtr=False)
            with self._port_lock:
                if self._closed.is_set():
                    port.close()
                    raise ProximityError("The proximity source is closed.")
                self._port = port
            port.port = self.settings.port
            port.dtr = False
            port.rts = False
            port.open()
            if self._closed.is_set():
                raise ProximityError("The proximity source is closed.")
            self._hello()
        except ProximityError:
            self.close()
            raise
        except (ImportError, OSError, ValueError, AttributeError) as exc:
            self.close()
            if isinstance(exc, ImportError):
                raise ProximityError("Serial proximity requires the optional pyserial package.") from exc
            raise ProximityError("The serial proximity source could not be opened or identified.") from exc

    def _token(self) -> str:
        token = self._token_factory()
        if not isinstance(token, str) or not _HEX16.fullmatch(token):
            raise ProximityError("The proximity request token is invalid.")
        return token

    def _exchange(self, command: str, kind: str, timeout_s: float,
                  *, overall_deadline_ns: int | None = None) -> tuple[dict, int, int]:
        token = self._token()
        sent_ns = time.monotonic_ns()
        deadline_ns = sent_ns + int(timeout_s * 1_000_000_000)
        if overall_deadline_ns is not None:
            deadline_ns = min(deadline_ns, overall_deadline_ns)
        if sent_ns >= deadline_ns:
            raise _ResponseTimeout("The sensor response deadline expired.")
        port = self._port
        if port is None or self._closed.is_set():
            raise ProximityError("The proximity source is closed.")
        request = f"{command} {token}\n".encode("ascii")
        if port.write(request) != len(request):
            raise ProximityError("The sensor request could not be written completely.")
        line = bytearray()
        discarded = False
        total = 0
        while time.monotonic_ns() < deadline_ns and not self._closed.is_set():
            chunk = port.read(1)
            if not chunk:
                continue
            if not isinstance(chunk, bytes) or len(chunk) != 1:
                raise ProximityError("The serial proximity stream is malformed.")
            total += 1
            if total > _MAX_EXCHANGE_BYTES:
                raise ProximityError("The serial proximity stream exceeded its input limit.")
            if chunk != b"\n":
                if len(line) < MAX_LINE_BYTES - 1 and not discarded:
                    line.extend(chunk)
                else:
                    discarded = True
                continue
            if discarded:
                raise ProximityError("The serial proximity response exceeded its line limit.")
            try:
                message = json.loads(line.decode("ascii"), object_pairs_hook=_unique_pairs,
                                     parse_constant=_no_constant)
            except (UnicodeError, ValueError, TypeError):
                line.clear()
                # Boot ROM noise is expected only before protocol identification.
                if kind == "hello":
                    continue
                raise ProximityError("The sensor response is not valid protocol JSON.") from None
            line.clear()
            if not isinstance(message, dict) or message.get("id") != token:
                continue  # bounded startup noise or old buffered requests
            if type(message.get("v")) is not int or message["v"] != 1:
                raise ProximityError("The sensor protocol version is incompatible.")
            if message.get("type") == "error":
                raise ProximityError("The sensor rejected the acquisition request.")
            if message.get("type") != kind:
                raise ProximityError("The sensor response type is incompatible.")
            received_ns = time.monotonic_ns()
            if received_ns >= deadline_ns:
                break
            return message, sent_ns, received_ns
        if self._closed.is_set():
            raise ProximityError("The proximity source is closed.")
        raise _ResponseTimeout("The sensor response deadline expired.")

    def _hello(self) -> None:
        # Opening some USB carriers resets the ESP32. An initial command can
        # disappear before Serial.begin; retry identification, never acquisition.
        deadline_ns = time.monotonic_ns() + int(self.settings.startup_timeout_s * 1_000_000_000)
        while not self._closed.is_set():
            remaining_s = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining_s <= 0:
                break
            try:
                response, _, _ = self._exchange(
                    "H1", "hello", min(.3, remaining_s), overall_deadline_ns=deadline_ns,
                )
            except _ResponseTimeout:
                continue
            boot = response.get("boot")
            if (response.get("fw") != "vidi_ultrasonic" or not isinstance(boot, str)
                    or not _HEX8.fullmatch(boot) or response.get("channels") != ["left", "right"]):
                raise ProximityError("The sensor firmware identity is incompatible.")
            self._boot = boot
            self._last_seq = None
            return
        if self._closed.is_set():
            raise ProximityError("The proximity source is closed.")
        raise ProximityError("The sensor did not identify protocol v1; flash the vidi_ultrasonic firmware and reconnect.")

    def sample(self) -> ProximitySnapshot:
        try:
            self._open()
            # Apply the same rate bound to direct diagnostic callers as monitor.
            interval_ns = int(max(.2, 1. / self.settings.poll_hz) * 1_000_000_000)
            if self._last_start_ns is not None:
                delay = (self._last_start_ns + interval_ns - time.monotonic_ns()) / 1_000_000_000
                if delay > 0 and self._closed.wait(delay):
                    raise ProximityError("The proximity source is closed.")
            response, sent_ns, received_ns = self._exchange("R1", "sample", self.settings.response_timeout_s)
            self._last_start_ns = sent_ns
            if response.get("boot") != self._boot:
                raise ProximityError("The sensor restarted; reconnect is required.")
            seq = response.get("seq")
            if type(seq) is not int or not 0 <= seq <= 0xFFFFFFFF:
                raise ProximityError("The sensor sequence number is malformed.")
            if self._last_seq is not None and not 0 < ((seq - self._last_seq) & 0xFFFFFFFF) < 0x80000000:
                raise ProximityError("The sensor response is duplicated or out of order.")
            observations = []
            max_age_ms = math.ceil(self.settings.stale_after_s * 1000)
            for name in ("left", "right"):
                value = response.get(name)
                if not isinstance(value, dict) or not {"status", "distance_mm", "age_ms"}.issubset(value):
                    raise ProximityError("A sensor observation is incomplete.")
                status, distance, age = validate_reading(value)
                if age > max_age_ms:
                    raise ProximityError("The sensor reported a stale acquisition.")
                observations.append(ProximityObservation(
                    name, status, distance, sent_ns, received_ns, age, self._boot, seq,
                ))
            self._last_seq = seq
            return ProximitySnapshot(*observations)
        except ProximityError:
            self.close()
            raise
        except KeyboardInterrupt:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ProximityError("The serial proximity acquisition failed.") from exc

    def close(self) -> None:
        """Cancel bounded reads and close idempotently, even after setup errors."""
        self._closed.set()
        with self._port_lock:
            port, self._port = self._port, None
        if port is not None:
            for method in ("cancel_read", "cancel_write", "close"):
                try:
                    callback = getattr(port, method, None)
                    if callback is not None:
                        callback()
                except Exception:
                    pass
