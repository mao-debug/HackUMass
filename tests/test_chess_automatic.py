"""Strict pair commits use ordinary legal history, with no camera/model calls."""

from dataclasses import replace
import threading

import pytest

from vidi.assistant.chess_automatic import AutomaticChessSession
from vidi.core.chess_types import BoardObservation

chess = pytest.importorskip("chess")
BASE = 10_000_000_000


def pair(board, number=0, **changes):
    squares = tuple(piece.symbol() if (piece := board.piece_at(square)) else "." for square in range(64))
    first = BoardObservation(number * 2 + 1, BASE + number * 1_000_000_000,
                             status="ready", squares=squares, **changes)
    return first, replace(first, frame_id=first.frame_id + 1, captured_monotonic_ns=first.captured_monotonic_ns + 300_000_000)


def apply(session, board, number=0, **kwargs):
    views = pair(board, number)
    return session.apply_pair(*views, now_ns=views[1].captured_monotonic_ns + 1, **kwargs)


def test_opt_in_disabled_by_default():
    session = AutomaticChessSession()
    assert apply(session, chess.Board()).status == "disabled"
    assert not session.session.initialized


def test_disabled_automatic_wrapper_preserves_existing_manual_age_limit():
    assert AutomaticChessSession(enabled=False, max_age_s=120).session.max_age_s == 120


def test_standard_start_and_each_legal_halfmove_commit_once():
    session = AutomaticChessSession(enabled=True)
    board = chess.Board()
    assert apply(session, board).status == "confirmed"
    board.push_uci("e2e4")
    result = apply(session, board, 1)
    assert result.status == "confirmed" and result.uci == "e2e4"
    assert session.snapshot()["confirmed_moves"] == ["e2e4"]
    assert apply(session, board, 1).status == "stale"


@pytest.mark.parametrize("sequence,expected", [
    (("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"), "castling kingside"),
    (("e2e4", "a7a6", "e4e5", "d7d5", "e5d6"), "en passant on d5"),
    (("a2a4", "b7b5", "a4b5", "a7a6", "b5a6", "h7h6", "a6a7", "h6h5", "a7b8n"), "promotes to knight"),
])
def test_special_moves_require_complete_standard_history(sequence, expected):
    session = AutomaticChessSession(enabled=True)
    board = chess.Board()
    apply(session, board)
    for number, move in enumerate(sequence, 1):
        board.push_uci(move)
        result = apply(session, board, number)
        assert result.status == "confirmed"
        assert session.snapshot()["fen"] == board.fen()
    assert expected in result.message


@pytest.mark.parametrize("change,status", [
    ({"status": "cancelled"}, "invalid_observation"),
    ({"status": "unclear"}, "invalid_observation"),
    ({"orientation": "black_near"}, "invalid_observation"),
    ({"squares": ("?",) * 64}, "invalid_observation"),
    ({"squares": (".",) * 64}, "disagreement"),
    ({"frame_id": 1}, "stale"),
    ({"captured_monotonic_ns": BASE + 299_000_000}, "stale"),
    ({"simulated": True}, "invalid_observation"),
])
def test_invalid_disagreeing_or_non_distinct_pair_never_commits(change, status):
    session = AutomaticChessSession(enabled=True)
    first, second = pair(chess.Board())
    result = session.apply_pair(first, replace(second, **change), now_ns=BASE + 400_000_000)
    assert result.status == status
    assert session.snapshot()["fen"] is None and session.session.pending is None


def test_expired_future_and_reused_first_frame_never_commit():
    session = AutomaticChessSession(enabled=True)
    first, second = pair(chess.Board())
    assert session.apply_pair(first, second, now_ns=BASE + 31_000_000_000).status == "stale"
    assert session.apply_pair(first, second, now_ns=BASE - 1).status == "stale"
    assert session.apply_pair(first, second, now_ns=BASE + 400_000_000).status == "confirmed"
    assert session.apply_pair(first, replace(second, frame_id=10, captured_monotonic_ns=BASE + 500_000_000), now_ns=BASE + 600_000_000).status == "stale"


def test_cancel_event_and_generation_reject_late_results_without_history_change():
    session = AutomaticChessSession(enabled=True)
    generation = session.generation
    session.cancel()
    assert apply(session, chess.Board(), generation=generation).status == "cancelled"
    event = threading.Event()
    event.set()
    assert apply(session, chess.Board(), cancel_event=event).status == "cancelled"
    assert not session.session.initialized


def test_wake_invalidation_can_preserve_manual_proposal_for_confirmation():
    session = AutomaticChessSession(enabled=True)
    first, second = pair(chess.Board())
    proposal = session.session.propose(second, now_ns=second.captured_monotonic_ns + 1)
    generation = session.generation
    session.cancel(discard_pending=False)
    assert session.generation == generation + 1
    assert session.session.pending.proposal_id == proposal.proposal_id
    assert session.session.confirm(proposal.proposal_id, now_ns=second.captured_monotonic_ns + 2).status == "confirmed"
    assert session.session.initialized


def test_cancel_between_propose_and_confirm_discards_proposal(monkeypatch):
    session = AutomaticChessSession(enabled=True)
    event = threading.Event()
    original = session.session.propose
    def propose(*args, **kwargs):
        result = original(*args, **kwargs)
        event.set()
        return result
    monkeypatch.setattr(session.session, "propose", propose)
    assert apply(session, chess.Board(), cancel_event=event).status == "cancelled"
    assert session.session.pending is None and not session.session.initialized


def test_illegal_or_skipped_moves_and_midgame_initialization_abstain():
    session = AutomaticChessSession(enabled=True)
    board = chess.Board()
    board.push_uci("e2e4")
    assert apply(session, board).status == "inconsistent"
    board = chess.Board()
    apply(session, board, 1)
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert apply(session, board, 2).status == "inconsistent"
    assert session.snapshot()["confirmed_moves"] == []


def test_simulated_pair_remains_explicitly_simulated_in_result_and_snapshot():
    session = AutomaticChessSession(enabled=True)
    views = pair(chess.Board(), simulated=True)
    result = session.apply_pair(*views, now_ns=views[1].captured_monotonic_ns + 1)
    assert result.status == "confirmed" and result.message.startswith("SIMULATED")
    assert session.snapshot()["simulated"] is True


@pytest.mark.parametrize("kwargs", [{"enabled": "yes"}, {"enabled": True, "max_age_s": 31}, {"max_age_s": 121}, {"max_age_s": True}, {"max_age_s": float("inf")}, {"max_age_s": 10**1000}])
def test_automatic_settings_are_explicit_and_bounded(kwargs):
    with pytest.raises(ValueError):
        AutomaticChessSession(**kwargs)
