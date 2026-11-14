"""Portable fixture, attribution, and monitor lifecycle tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json
import threading
import time
from types import SimpleNamespace

import pytest

from vidi.sensors import proximity
from vidi.sensors.fixture import FixtureSource
from vidi.sensors.proximity import (
    ProximityError, ProximityMonitor, ProximityObservation, ProximitySnapshot,
    create_source, format_proximity,
)


def settings(**overrides):
    values = dict(backend="fixture", port=None, fixture_path=None, baud_rate=115200,
                  poll_hz=5., response_timeout_s=.4, stale_after_s=.5,
                  nearby_mm=1000, startup_timeout_s=3.)
    values.update(overrides)
    return SimpleNamespace(**values)


def snapshot(*, distance=500, status="ok", sent=None, simulated=False):
    sent = time.monotonic_ns() if sent is None else sent
    return ProximitySnapshot(*(
        ProximityObservation(name, status, distance if status == "ok" else None,
                             sent, sent, 0, "abcdef12", 1, simulated)
        for name in ("left", "right")
    ))


def write_fixture(tmp_path, document=None):
    path = tmp_path / "proximity.json"
    path.write_text(json.dumps(document or {"v": 1, "snapshots": [
        {"left": {"status": "ok", "distance_mm": 500},
         "right": {"status": "no_echo", "distance_mm": None}},
        {"left": {"status": "out_of_range", "distance_mm": None},
         "right": {"status": "ok", "distance_mm": 2000}},
    ]}))
    return path


def test_fixture_advances_repeats_last_and_is_explicitly_simulated(tmp_path):
    source = FixtureSource(settings(fixture_path=write_fixture(tmp_path)))
    first, second, third = source.sample(), source.sample(), source.sample()
    assert first.left.distance_mm == 500 and first.right.status == "no_echo"
    assert second.left.status == "out_of_range" and second.right.distance_mm == 2000
    assert third.right.distance_mm == second.right.distance_mm
    assert third.left.seq == 2 and first.simulated
    assert first.left.request_sent_monotonic_ns <= first.left.received_monotonic_ns
    assert first.to_dict() == asdict(first)
    with pytest.raises(FrozenInstanceError):
        first.left.distance_mm = 3
    source.close()
    source.close()
    with pytest.raises(ProximityError, match="closed"):
        source.sample()


@pytest.mark.parametrize("document", [
    {"v": True, "snapshots": []}, {"v": 2, "snapshots": []},
    {"v": 1, "snapshots": []}, {"v": 1, "snapshots": [{}]},
    {"v": 1, "snapshots": [{"left": {}, "right": {}}]},
    {"v": 1, "snapshots": [{"left": {"status": [], "distance_mm": None}, "right": {}}]},
])
def test_malformed_fixture_fails_without_any_hardware(tmp_path, document):
    with pytest.raises(ProximityError):
        FixtureSource(settings(fixture_path=write_fixture(tmp_path, document)))


def test_disabled_and_fixture_sources_do_not_import_hardware_dependencies(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__

    def restricted(name, *args, **kwargs):
        assert name.split(".")[0] not in {"serial", "picamera2", "libcamera", "cv2", "google"}
        return original(name, *args, **kwargs)

    path = write_fixture(tmp_path)
    monkeypatch.setattr(builtins, "__import__", restricted)
    assert create_source(settings(backend="disabled")).sample().left.status == "disabled"
    assert create_source(settings(fixture_path=path)).sample().simulated


def test_fixture_file_size_and_read_errors_are_bounded(tmp_path):
    path = tmp_path / "huge.json"
    path.write_bytes(b" " * 65537)
    with pytest.raises(ProximityError, match="large"):
        FixtureSource(settings(fixture_path=path))
    with pytest.raises(ProximityError):
        FixtureSource(settings(fixture_path=tmp_path / "missing.json"))


def test_formatter_independently_attributes_unknown_and_nearby_returns():
    pair = snapshot(simulated=True)
    pair = ProximitySnapshot(pair.left, ProximityObservation("right", "no_echo", simulated=True))
    text = format_proximity(pair)
    assert text == "Simulated proximity. Left sensor: nearby return. Right sensor: reading unavailable."
    assert all(term not in text.lower() for term in ("clear", "center", "chair", "safe"))
    assert "beyond the nearby threshold" in format_proximity(snapshot(distance=1500))


@pytest.mark.parametrize("status", ["no_echo", "out_of_range", "error", "stale", "unavailable", "disabled"])
def test_unknown_states_are_never_spoken_as_clear(status):
    text = format_proximity(snapshot(status=status))
    assert text.count("reading unavailable") == 2 and "clear" not in text.lower()


def test_latest_uses_request_age_not_recent_receipt_and_preserves_metadata(monkeypatch):
    monitor = ProximityMonitor(settings())
    old = snapshot(sent=1_000_000_000)
    monitor._snapshot = ProximitySnapshot(
        old.left, ProximityObservation("right", "ok", 600, 1_000_000_000,
                                     1_599_000_000, 0, "abcdef12", 5),
    )
    monkeypatch.setattr(proximity.time, "monotonic_ns", lambda: 1_600_000_000)
    result = monitor.latest()
    assert [item.status for item in result.observations] == ["stale", "stale"]
    assert all(item.distance_mm is None for item in result.observations)
    assert result.right.seq == 5 and result.right.received_monotonic_ns == 1_599_000_000


def test_reported_duration_cannot_be_promoted_to_fresh_simulated_data():
    monitor = ProximityMonitor(settings())
    now = time.monotonic_ns()
    monitor._snapshot = ProximitySnapshot(*(
        ProximityObservation(name, "ok", 500, now, now, 600, simulated=True)
        for name in ("left", "right")
    ))
    assert monitor.latest().left.status == "stale"


def test_public_refresh_reages_exact_action_without_consulting_monitor_cache():
    result = proximity.refresh_snapshot(snapshot(sent=1_000_000_000), .5, now_ns=1_500_000_001)
    assert result.left.status == "stale" and result.left.distance_mm is None
    assert result.left.request_sent_monotonic_ns == 1_000_000_000


class ImmediateSource:
    def __init__(self):
        self.closed = threading.Event()
        self.starts = []

    def sample(self):
        self.starts.append(time.monotonic_ns())
        return snapshot(simulated=True)

    def close(self):
        self.closed.set()


def test_monitor_new_requests_are_fresh_rate_bounded_and_closed():
    source = ImmediateSource()
    checkpoint = time.monotonic_ns()
    with ProximityMonitor(settings(poll_hz=100.), source_factory=lambda _: source) as monitor:
        first = monitor.fresh(after_ns=checkpoint)
        assert first.simulated and first.left.request_sent_monotonic_ns > checkpoint
        checkpoint = time.monotonic_ns()
        second = monitor.fresh(after_ns=checkpoint)
        assert second.left.request_sent_monotonic_ns > checkpoint
        assert len(source.starts) == 2
        assert source.starts[1] - source.starts[0] >= 195_000_000
    assert source.closed.is_set()
    monitor.close()


def test_fresh_does_not_return_a_cached_pre_checkpoint_pair():
    monitor = ProximityMonitor(settings())
    monitor._snapshot = snapshot()
    current = time.monotonic_ns()
    result = monitor.fresh(after_ns=current, timeout_s=.01)
    assert result.left.status == "unknown" and result.left.distance_mm is None
    cancel = threading.Event()
    cancel.set()
    assert monitor.fresh(current, cancel_event=cancel).left.status == "cancelled"
    monitor.close()
    assert monitor.fresh(current).left.distance_mm is None


def test_monitor_pauses_on_failure_and_only_reconnect_opens_replacement():
    failed = ImmediateSource()
    failed.sample = lambda: (_ for _ in ()).throw(OSError("private-secret"))
    replacement = ImmediateSource()
    calls = []

    def factory(_settings):
        calls.append(1)
        return failed if len(calls) == 1 else replacement

    with ProximityMonitor(settings(), source_factory=factory) as monitor:
        result = monitor.fresh(time.monotonic_ns())
        assert result.left.status == "unavailable" and monitor.paused
        assert "private-secret" not in result.left.error
        assert failed.closed.is_set() and len(calls) == 1
        checkpoint = time.monotonic_ns()
        monitor.reconnect()
        assert monitor.latest().left.distance_mm is None
        assert monitor.fresh(checkpoint).left.status == "ok"
        assert not monitor.paused and len(calls) == 2
    assert replacement.closed.is_set()


def test_preview_access_and_shutdown_remain_bounded_during_a_blocked_read():
    entered, release = threading.Event(), threading.Event()

    class Blocked:
        def sample(self):
            entered.set()
            release.wait(2)
            raise ProximityError("cancelled")

        def close(self):
            release.set()

    monitor = ProximityMonitor(settings(), source_factory=lambda _: Blocked())
    monitor.__enter__()
    assert entered.wait(.5)
    started = time.monotonic()
    assert monitor.latest().left.status == "unknown"
    monitor.close()
    assert time.monotonic() - started < .5
    assert release.is_set() and not monitor._thread.is_alive()


def test_cleanup_failure_cannot_crash_the_monitor_or_hide_unavailable_state():
    class Broken:
        def sample(self):
            raise OSError("device gone")

        def close(self):
            raise OSError("close failed")

    with ProximityMonitor(settings(), source_factory=lambda _: Broken()) as monitor:
        assert monitor.fresh(time.monotonic_ns()).left.status == "unavailable"
        assert monitor.paused


def test_known_safe_error_keeps_actionable_reason_for_reconnect():
    source = ImmediateSource()
    source.sample = lambda: (_ for _ in ()).throw(ProximityError("Protocol v1 firmware is required."))
    with ProximityMonitor(settings(), source_factory=lambda _: source) as monitor:
        result = monitor.fresh(time.monotonic_ns())
        assert result.left.error == "Protocol v1 firmware is required."


def test_disabled_monitor_does_not_start_a_worker_or_open_a_source():
    with ProximityMonitor(settings(backend="disabled"), source_factory=lambda _: pytest.fail("opened")) as monitor:
        assert monitor._thread is None
        assert monitor.fresh(time.monotonic_ns()).left.status == "disabled"
