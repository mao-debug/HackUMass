"""Protocol tests use in-memory serial transport, never USB devices."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vidi.sensors import serial_source
from vidi.sensors.proximity import ProximityError


def settings(**overrides):
    values = dict(backend="serial", port="/dev/serial/by-id/explicit-board", fixture_path=None,
                  baud_rate=115200, poll_hz=5., response_timeout_s=.4,
                  stale_after_s=.5, nearby_mm=1000, startup_timeout_s=3.)
    values.update(overrides)
    return SimpleNamespace(**values)


class Clock:
    now = 1_000_000_000

    def advance(self, seconds):
        self.now += int(seconds * 1_000_000_000)

    def monotonic_ns(self):
        return self.now


class FakePort:
    def __init__(self, clock, **kwargs):
        self.clock = clock
        self.kwargs = kwargs
        self.data = bytearray()
        self.writes = []
        self.opened = False
        self.closed = False
        self.next_seq = 1
        self.mutate = lambda command, value: value

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def write(self, data):
        self.writes.append(data)
        command, token = data.decode().strip().split()
        response = dict(v=1, type="hello", id=token, fw="vidi_ultrasonic",
                        boot="abcdef12", channels=["left", "right"])
        if command == "R1":
            response = dict(v=1, type="sample", id=token, boot="abcdef12", seq=self.next_seq,
                            left=dict(status="ok", distance_mm=750, age_ms=100),
                            right=dict(status="no_echo", distance_mm=None, age_ms=0))
            self.next_seq = (self.next_seq + 1) & 0xFFFFFFFF
        changed = self.mutate(command, response)
        self.data.extend(changed if isinstance(changed, bytes) else (json.dumps(changed) + "\n").encode())
        return len(data)

    def read(self, _size):
        self.clock.advance(.00001 if self.data else .01)
        if not self.data:
            return b""
        value = bytes(self.data[:1])
        del self.data[:1]
        return value


@pytest.fixture
def transport(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(serial_source.time, "monotonic_ns", clock.monotonic_ns)
    port = FakePort(clock)
    counter = iter(range(1000))
    source = serial_source.SerialSource(settings(), serial_factory=lambda **kw: _capture(port, kw),
                                         token_factory=lambda: f"{next(counter):016x}")
    monkeypatch.setattr(source._closed, "wait", lambda seconds: clock.advance(seconds) or False)
    return source, port, clock


def _capture(port, kwargs):
    port.kwargs = kwargs
    return port


def test_matching_new_request_preserves_channel_status_and_host_clock(transport):
    source, port, clock = transport
    result = source.sample()
    assert [item.split()[0] for item in port.writes] == [b"H1", b"R1"]
    assert port.writes[0] != port.writes[1]
    assert port.port == "/dev/serial/by-id/explicit-board"
    assert port.kwargs["port"] is None and not port.dtr and not port.rts
    assert not port.kwargs["rtscts"] and not port.kwargs["dsrdtr"]
    assert result.left.distance_mm == 750 and result.right.status == "no_echo"
    assert result.right.distance_mm is None and not result.simulated
    assert result.left.request_sent_monotonic_ns <= result.left.received_monotonic_ns <= clock.now
    assert result.left.reported_age_ms == 100
    source.close()
    source.close()
    assert port.closed


def test_boot_noise_and_unmatched_buffered_responses_cannot_become_samples(transport):
    source, port, _ = transport

    def mutate(command, response):
        old = {**response, "id": "ffffffffffffffff"}
        prefix = b"ROM boot text\n" if command == "H1" else b""
        return prefix + (json.dumps(old) + "\n" + json.dumps(response) + "\n").encode()

    port.mutate = mutate
    assert source.sample().left.seq == 1


def test_hello_retries_with_new_token_when_carrier_reset_discards_first_request(transport):
    source, port, clock = transport
    attempts = []

    def mutate(command, response):
        if command == "H1":
            attempts.append(clock.now)
            if len(attempts) == 1:
                return b"ROM booting\n"
        return response

    port.mutate = mutate
    result = source.sample()
    assert result.left.distance_mm == 750
    assert [item.split()[0] for item in port.writes] == [b"H1", b"H1", b"R1"]
    assert port.writes[0] != port.writes[1]
    assert attempts[1] - attempts[0] >= 200_000_000


def test_no_hello_response_has_one_overall_startup_deadline(transport):
    source, port, clock = transport
    port.mutate = lambda command, value: b""
    started = clock.now
    with pytest.raises(ProximityError, match="protocol v1"):
        source.sample()
    elapsed = clock.now - started
    assert 3_000_000_000 <= elapsed <= 3_020_000_000
    assert 2 <= len(port.writes) <= 11
    assert all(item.startswith(b"H1 ") for item in port.writes)
    assert port.closed


def test_identified_wrong_hello_version_fails_without_retry(transport):
    source, port, _ = transport
    port.mutate = lambda command, value: {**value, "v": 2}
    with pytest.raises(ProximityError, match="version"):
        source.sample()
    assert len(port.writes) == 1 and port.closed


@pytest.mark.parametrize("change", [
    lambda v: {**v, "boot": "12345678"},
    lambda v: {**v, "v": True},
    lambda v: {**v, "v": 2},
    lambda v: {**v, "seq": True},
    lambda v: {**v, "seq": -1},
    lambda v: {**v, "seq": 2**32},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": 19, "age_ms": 0}},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": 3001, "age_ms": 0}},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": True, "age_ms": 0}},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": 750.0, "age_ms": 0}},
    lambda v: {**v, "left": {"status": "no_echo", "distance_mm": 750, "age_ms": 0}},
    lambda v: {**v, "left": {"status": "coasting", "distance_mm": None, "age_ms": 0}},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": 750, "age_ms": -1}},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": 750, "age_ms": True}},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": 750, "age_ms": 501}},
    lambda v: {**v, "left": {"status": "ok", "distance_mm": 750}},
    lambda v: {**v, "type": "error", "code": "busy"},
])
def test_matching_invalid_response_fails_closed_and_releases_port(transport, change):
    source, port, _ = transport
    port.mutate = lambda command, value: change(value) if command == "R1" else value
    with pytest.raises(ProximityError):
        source.sample()
    assert port.closed


@pytest.mark.parametrize("change", [
    lambda v: {**v, "fw": "other_firmware"},
    lambda v: {**v, "boot": "ABCDEF12"},
    lambda v: {**v, "channels": ["right", "left"]},
])
def test_firmware_identity_must_be_compatible(transport, change):
    source, port, _ = transport
    port.mutate = lambda command, value: change(value)
    with pytest.raises(ProximityError):
        source.sample()
    assert port.closed and len(port.writes) == 1


def test_duplicate_sequence_rejected_but_uint32_rollover_accepted(transport):
    source, port, _ = transport
    port.next_seq = 0xFFFFFFFF
    assert source.sample().left.seq == 0xFFFFFFFF
    assert source.sample().left.seq == 0
    port.next_seq = 0
    with pytest.raises(ProximityError, match="duplicated"):
        source.sample()
    assert port.closed


def test_direct_callers_observe_maximum_five_requests_per_second(transport):
    source, port, _ = transport
    starts = []

    def mutate(command, value):
        if command == "R1":
            starts.append(port.clock.now)
        return value

    port.mutate = mutate
    source.sample()
    source.sample()
    source.sample()
    assert all(b - a >= 200_000_000 for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize("payload", [b"not json\n", b"x" * 512 + b"\n", b'{"v":1,"v":1}\n'])
def test_malformed_or_oversized_sample_stream_is_rejected(transport, payload):
    source, port, _ = transport
    port.mutate = lambda command, value: payload if command == "R1" else value
    with pytest.raises(ProximityError):
        source.sample()
    assert port.closed


def test_absent_response_expires_and_discards_old_token(transport):
    source, port, clock = transport
    port.mutate = lambda command, value: {**value, "id": "eeeeeeeeeeeeeeee"} if command == "R1" else value
    started = clock.now
    with pytest.raises(ProximityError, match="deadline"):
        source.sample()
    assert clock.now - started < 500_000_000 and port.closed
    assert [item.split()[0] for item in port.writes] == [b"H1", b"R1"]


def test_open_failure_is_sanitized_and_closed(transport):
    source, port, _ = transport
    port.open = lambda: (_ for _ in ()).throw(OSError("private-device-path SECRET"))
    with pytest.raises(ProximityError) as caught:
        source.sample()
    assert "SECRET" not in str(caught.value) and port.closed


def test_pyserial_missing_is_optional_and_controlled(monkeypatch):
    import builtins
    original = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name == "serial":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted)
    source = serial_source.SerialSource(settings())
    with pytest.raises(ProximityError, match="optional pyserial"):
        source.sample()
