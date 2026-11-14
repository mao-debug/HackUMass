"""Independent local proximity observations; no camera or cloud imports."""

from .proximity import (
    ProximityError,
    ProximityMonitor,
    ProximityObservation,
    ProximitySnapshot,
    create_source,
    format_proximity,
    refresh_snapshot,
)

__all__ = [
    "ProximityError", "ProximityMonitor", "ProximityObservation",
    "ProximitySnapshot", "create_source", "format_proximity", "refresh_snapshot",
]
