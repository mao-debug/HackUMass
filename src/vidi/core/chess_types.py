"""Camera-linked chess observations without importing optional chess rules."""

from dataclasses import dataclass


SQUARE_NAMES = tuple(f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh")
PIECE_SYMBOLS = frozenset("PNBRQKpnbrqk.?")


@dataclass(frozen=True)
class BoardObservation:
    """One visual observation; squares run a1..h1, a2..h2, through a8..h8.

    Uppercase symbols mean white pieces, lowercase black; '.' is observed empty
    and '?' is unknown. The rules session must independently validate an
    observation before proposing any state change.
    """

    frame_id: int
    captured_monotonic_ns: int
    status: str = "unclear"
    squares: tuple[str, ...] = ()
    orientation: str = "white_near"
    model: str | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    simulated: bool = False
