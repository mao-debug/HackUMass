"""Explicitly enabled, conservative two-view commits through normal chess rules.

This is software validation, not proof that two model predictions are true.
The default session remains human-confirmed. Cancellation generations let the
runtime discard a result from a scan that was stopped while work was pending.
"""

from __future__ import annotations

import math
import threading
import time

from vidi.assistant.chess_session import ChessSession, Transition
from vidi.core.chess_types import BoardObservation, PIECE_SYMBOLS


class AutomaticChessSession:
    """Commit only identical, complete, fresh observations of distinct frames."""

    def __init__(self, *, enabled=False, orientation="white_near", max_age_s=30):
        if type(enabled) is not bool:
            raise ValueError("Automatic chess requires an explicit boolean opt-in.")
        try:
            valid_age = (type(max_age_s) in (int, float) and math.isfinite(max_age_s)
                         and 0 < max_age_s <= (30 if enabled else 120))
        except OverflowError:
            valid_age = False
        if not valid_age:
            raise ValueError("Chess observations must expire within 30 seconds for automation or 120 seconds for manual mode.")
        self.enabled = enabled
        self.session = ChessSession(orientation=orientation, max_age_s=max_age_s)
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def cancel(self, *, discard_pending=True):
        """Invalidate work; wake activation may preserve a manual proposal."""
        if type(discard_pending) is not bool:
            raise ValueError("discard_pending must be a boolean.")
        with self._lock:
            self._generation += 1
            if discard_pending:
                self.session.reject()

    def apply_pair(self, first: BoardObservation, second: BoardObservation, *,
                   cancel_event: threading.Event | None = None,
                   now_ns: int | None = None, generation: int | None = None) -> Transition:
        """Validate both views before proposing and immediately confirming one step."""
        clock_is_live = now_ns is None
        now_ns = time.monotonic_ns() if clock_is_live else now_ns
        with self._lock:
            initial_generation = self._generation
            if not self.enabled:
                return Transition("disabled", "Automatic chess is disabled. Use separate scan and confirmation controls.")
            if ((cancel_event is not None and cancel_event.is_set())
                    or (generation is not None and (type(generation) is not int or generation != self._generation))):
                return Transition("cancelled", "Cancelled board observations were discarded. Confirmed history is unchanged.")
            if self.session.pending is not None:
                return Transition("pending", "A separate proposal is pending. Confirm or reject it before automatic tracking.")
            if (not all(isinstance(item, BoardObservation) for item in (first, second))
                    or any(item.status != "ready" or item.orientation != self.session.orientation
                           or not isinstance(item.squares, tuple) or len(item.squares) != 64
                           or any(type(piece) is not str or piece not in PIECE_SYMBOLS or piece == "?" for piece in item.squares)
                           or type(item.simulated) is not bool for item in (first, second))):
                return Transition("invalid_observation", "Both complete board views must be verified before automatic tracking.")
            if first.simulated != second.simulated:
                return Transition("invalid_observation", "Simulated and physical board observations cannot be mixed.")
            if first.squares != second.squares:
                return Transition("disagreement", "The two board views disagree. Confirmed history is unchanged.")
            accepted = self.session.snapshot()
            if (type(first.frame_id) is not int or type(second.frame_id) is not int
                    or not 0 < first.frame_id < second.frame_id
                    or type(first.captured_monotonic_ns) is not int or type(second.captured_monotonic_ns) is not int
                    or second.captured_monotonic_ns - first.captured_monotonic_ns < 300_000_000
                    or not self.session._fresh(first.captured_monotonic_ns, now_ns)
                    or not self.session._fresh(second.captured_monotonic_ns, now_ns)
                    or first.frame_id <= accepted["accepted_frame_id"]
                    or first.captured_monotonic_ns <= accepted["accepted_captured_monotonic_ns"]):
                return Transition("stale", "Use two fresh distinct board frames at least 300 milliseconds apart.")
            transition = self.session.propose(second, now_ns=now_ns)
            if transition.status != "proposed":
                if first.simulated:
                    return Transition(transition.status, "SIMULATED. " + transition.message,
                                      transition.proposal_id, transition.uci)
                return transition
            if ((cancel_event is not None and cancel_event.is_set())
                    or self._generation != initial_generation):
                self.session.reject()
                return Transition("cancelled", "Cancelled board observations were discarded. Confirmed history is unchanged.")
            commit_now = time.monotonic_ns() if clock_is_live else now_ns
            if not self.session._fresh(first.captured_monotonic_ns, commit_now):
                self.session.reject()
                return Transition("stale", "The board views expired before confirmation. Confirmed history is unchanged.")
            confirmed = self.session.confirm(transition.proposal_id, now_ns=commit_now)
            if first.simulated:
                return Transition(confirmed.status, "SIMULATED. " + confirmed.message,
                                  confirmed.proposal_id, confirmed.uci)
            return confirmed

    def snapshot(self):
        with self._lock:
            return {**self.session.snapshot(), "automatic": self.enabled, "generation": self._generation}
