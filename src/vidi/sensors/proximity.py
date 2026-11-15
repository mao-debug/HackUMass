"""Local, independently attributed proximity data and bounded background polling.

An observation's host request-start time is a conservative bound on its age:
the protocol requires acquisition AFTER that request. ESP32 timestamps are never
compared with the host clock. No distance is associated with a camera object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import threading
import time
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from vidi.core.configuration import ProximitySettings


class ProximityError(RuntimeError):
    """A local source could not provide a validated observation."""


@dataclass(frozen=True)
class ProximityObservation:
    sensor_id: str
    status: str
    distance_mm: int | None = None
    request_sent_monotonic_ns: int | None = None
    received_monotonic_ns: int | None = None
    reported_age_ms: int | None = None
    boot_id: str | None = None
    seq: int | None = None
    simulated: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ProximitySnapshot:
    left: ProximityObservation
    right: ProximityObservation

    @property
    def observations(self) -> tuple[ProximityObservation, ProximityObservation]:
        return self.left, self.right

    @property
    def simulated(self) -> bool:
        return any(item.simulated for item in self.observations)

    def to_dict(self) -> dict:
        """Return JSON-ready primitive values without driver/device objects."""
        return asdict(self)


def unknown_snapshot(
    status: str = "unknown", *, error: str | None = None, simulated: bool = False,
) -> ProximitySnapshot:
    return ProximitySnapshot(*(
        ProximityObservation(name, status, simulated=simulated, error=error)
        for name in ("left", "right")
    ))


def _freshness(
    snapshot: ProximitySnapshot, stale_after_s: float, now_ns: int,
) -> ProximitySnapshot:
    limit_ns = int(stale_after_s * 1_000_000_000)
    observations = []
    for item in snapshot.observations:
        sent, received = item.request_sent_monotonic_ns, item.received_monotonic_ns
        if sent is not None and received is not None:
            # Reported duration is an additional bound, not a device clock.
            age_ns = max(
                now_ns - sent,
                (item.reported_age_ms or 0) * 1_000_000 + now_ns - received,
            )
            if sent > received or received > now_ns or age_ns > limit_ns:
                item = replace(item, status="stale", distance_mm=None,
                               error="The sensor observation is stale.")
        elif item.status == "ok":
            item = replace(item, status="unknown", distance_mm=None,
                           error="The sensor observation has no acquisition time.")
        observations.append(item)
    return ProximitySnapshot(*observations)


def refresh_snapshot(
    snapshot: ProximitySnapshot, stale_after_s: float, now_ns: int | None = None,
) -> ProximitySnapshot:
    """Re-age an exact action result before speech without substituting a cache."""
    return _freshness(snapshot, stale_after_s, time.monotonic_ns() if now_ns is None else now_ns)


def format_proximity(snapshot: ProximitySnapshot, nearby_mm: int = 1000) -> str:
    """Describe each channel without claiming direction, objects, or a clear path."""
    phrases = []
    for item in snapshot.observations:
        label = "Left sensor" if item.sensor_id == "left" else "Right sensor"
        if item.status == "ok" and type(item.distance_mm) is int and 20 <= item.distance_mm <= 3000:
            reading = "nearby return" if item.distance_mm <= nearby_mm else "return beyond the nearby threshold"
        else:
            reading = "reading unavailable"
        phrases.append(f"{label}: {reading}.")
    prefix = "Simulated proximity. " if snapshot.simulated else ""
    return prefix + " ".join(phrases)


class ProximitySource(Protocol):
    def sample(self) -> ProximitySnapshot: ...
    def close(self) -> None: ...


class _DisabledSource:
    def sample(self) -> ProximitySnapshot:
        return unknown_snapshot("disabled")

    def close(self) -> None:
        pass


def create_source(settings: ProximitySettings) -> ProximitySource:
    """Construct a source without importing serial for disabled/fixture modes."""
    if settings.backend == "disabled":
        return _DisabledSource()
    if settings.backend == "fixture":
        from .fixture import FixtureSource
        return FixtureSource(settings)
    if settings.backend == "serial":
        from .serial_source import SerialSource
        return SerialSource(settings)
    raise ProximityError("The proximity backend is unsupported.")


class ProximityMonitor:
    """One daemon worker, no queued requests, and bounded caller-side shutdown.

    Runtime failures pause acquisition until explicit reconnect. A driver that
    fails to return from OS open/close cannot block the preview or interpreter
    shutdown; the daemon owns cleanup when that call eventually returns.
    """

    def __init__(self, settings: ProximitySettings, *,
                 source_factory: Callable = create_source) -> None:
        self.settings = settings
        self._source_factory = source_factory
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._paused = False
        self._source: ProximitySource | None = None
        self._thread: threading.Thread | None = None
        self._snapshot = unknown_snapshot(
            "disabled" if settings.backend == "disabled" else "unknown",
            simulated=settings.backend == "fixture",
        )

    def __enter__(self) -> ProximityMonitor:
        if self._thread is None and not self._stop.is_set() and self.settings.backend != "disabled":
            self._thread = threading.Thread(target=self._run, name="vidi-proximity", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    def latest(self) -> ProximitySnapshot:
        with self._condition:
            return _freshness(self._snapshot, self.settings.stale_after_s, time.monotonic_ns())

    def fresh(self, after_ns: int, timeout_s: float = .75,
              cancel_event: threading.Event | None = None) -> ProximitySnapshot:
        """Wait for a request started after a checkpoint, or return unknown."""
        deadline = time.monotonic() + max(0., timeout_s)
        with self._condition:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return unknown_snapshot("cancelled", simulated=self.settings.backend == "fixture")
                current = _freshness(self._snapshot, self.settings.stale_after_s, time.monotonic_ns())
                if all(item.request_sent_monotonic_ns is not None
                       and item.request_sent_monotonic_ns > after_ns
                       and item.status != "stale" for item in current.observations):
                    return current
                if self._stop.is_set():
                    return unknown_snapshot("cancelled", simulated=self.settings.backend == "fixture")
                if self._paused or self.settings.backend == "disabled":
                    return current
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return unknown_snapshot("unknown", error="A fresh sensor observation was not available in time.",
                                            simulated=self.settings.backend == "fixture")
                self._condition.wait(min(remaining, .05))

    def reconnect(self) -> None:
        """Ask the worker to replace its source; never open a device here."""
        if not self._stop.is_set():
            self._reconnect.set()
            self._publish(unknown_snapshot(simulated=self.settings.backend == "fixture"))

    def _publish(self, snapshot: ProximitySnapshot, *, paused: bool = False) -> None:
        with self._condition:
            self._snapshot = snapshot
            self._paused = paused
            self._condition.notify_all()

    def _run(self) -> None:
        interval = max(.2, 1. / self.settings.poll_hz)
        next_start = 0.
        try:
            while not self._stop.is_set():
                if self._reconnect.is_set():
                    self._reconnect.clear()
                    if self._source is not None:
                        self._close_source(self._source)
                    self._source = None
                    self._publish(unknown_snapshot(simulated=self.settings.backend == "fixture"))
                if self.paused:
                    self._stop.wait(.05)
                    continue
                delay = next_start - time.monotonic()
                if delay > 0:
                    self._stop.wait(min(delay, .05))
                    continue
                try:
                    if self._source is None:
                        self._source = self._source_factory(self.settings)
                    if self._stop.is_set():
                        break
                    snapshot = self._source.sample()
                    # A gap after completion excludes source setup and thread
                    # scheduling delays from the next request's spacing. This
                    # is a rate ceiling, not a promise of five pairs/second.
                    next_start = time.monotonic() + interval
                    if not self._stop.is_set():
                        self._publish(snapshot)
                except Exception as exc:
                    # Driver errors/paths and arbitrary input are not user text.
                    if self._source is not None:
                        self._close_source(self._source)
                    self._source = None
                    if not self._stop.is_set():
                        message = str(exc) if isinstance(exc, ProximityError) else "The proximity source is unavailable."
                        self._publish(unknown_snapshot("unavailable", error=message,
                                                      simulated=self.settings.backend == "fixture"), paused=True)
        finally:
            if self._source is not None:
                self._close_source(self._source)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        deadline = time.monotonic() + 1.
        source = self._source
        if source is not None:
            # Never run a possibly blocking driver close in the GUI thread.
            closer = threading.Thread(target=self._close_source, args=(source,), daemon=True)
            closer.start()
            closer.join(max(0., deadline - time.monotonic()))
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(max(0., deadline - time.monotonic()))

    @staticmethod
    def _close_source(source: ProximitySource) -> None:
        try:
            source.close()
        except Exception:
            pass
