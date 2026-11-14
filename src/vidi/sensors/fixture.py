"""Deterministic proximity fixtures without serial, GPIO, camera, or SDK imports."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from .proximity import ProximityError, ProximityObservation, ProximitySnapshot

if TYPE_CHECKING:
    from vidi.core.configuration import ProximitySettings

STATUSES = {"ok", "no_echo", "out_of_range", "error"}


def validate_reading(value: object) -> tuple[str, int | None, int]:
    """Strict shared observation schema; booleans/floats are not measurements."""
    if not isinstance(value, dict):
        raise ProximityError("A sensor observation is malformed.")
    status, distance, age = value.get("status"), value.get("distance_mm"), value.get("age_ms", 0)
    if not isinstance(status, str) or status not in STATUSES or type(age) is not int or not 0 <= age <= 0xFFFFFFFF:
        raise ProximityError("A sensor observation is malformed.")
    if status == "ok":
        if type(distance) is not int or not 20 <= distance <= 3000:
            raise ProximityError("A sensor measurement is outside the supported range.")
    elif distance is not None:
        raise ProximityError("An unavailable sensor observation must not contain a distance.")
    return status, distance, age


class FixtureSource:
    """Advance through snapshots, then repeat the last explicitly simulated pair.

    Schema: {"v":1,"snapshots":[{"left":{"status":"ok","distance_mm":500},
    "right":{"status":"no_echo","distance_mm":null}}]}.
    Optional age_ms permits deterministic stale-state simulation.
    """

    def __init__(self, settings: ProximitySettings) -> None:
        self._closed = False
        self._index = 0
        try:
            if settings.fixture_path is None:
                raise ProximityError("A proximity fixture path is required.")
            with settings.fixture_path.open("rb") as stream:
                payload = stream.read(65537)
            if len(payload) > 65536:
                raise ProximityError("The proximity fixture is too large.")
            document = json.loads(payload)
            if not isinstance(document, dict) or type(document.get("v")) is not int or document["v"] != 1:
                raise ProximityError("The proximity fixture version is unsupported.")
            samples = document.get("snapshots")
            if not isinstance(samples, list) or not 1 <= len(samples) <= 1000:
                raise ProximityError("The proximity fixture needs a bounded nonempty snapshots list.")
            self._samples = []
            for pair in samples:
                if not isinstance(pair, dict) or set(pair) != {"left", "right"}:
                    raise ProximityError("A proximity fixture must contain left and right observations.")
                self._samples.append(tuple(validate_reading(pair[name]) for name in ("left", "right")))
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            raise ProximityError("The proximity fixture could not be read or validated.") from exc

    def sample(self) -> ProximitySnapshot:
        if self._closed:
            raise ProximityError("The proximity source is closed.")
        now_ns = time.monotonic_ns()
        readings = self._samples[min(self._index, len(self._samples) - 1)]
        seq = self._index
        self._index += 1
        return ProximitySnapshot(*(
            ProximityObservation(name, status, distance, now_ns, now_ns, age, "fixture", seq, True)
            for name, (status, distance, age) in zip(("left", "right"), readings)
        ))

    def close(self) -> None:
        self._closed = True
