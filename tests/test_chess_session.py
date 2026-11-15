"""Camera boards become hypotheses; only explicit confirmation changes history."""

import builtins
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from vidi.assistant.chess_session import ChessSession, ChessSessionError
from vidi.core.chess_types import BoardObservation

chess = pytest.importorskip("chess", reason="Optional approved chess rules package is not installed")

BASE_NS = 1_000_000_000


def observation(board, frame=1, **changes):
    values = tuple(piece.symbol() if (piece := board.piece_at(square)) else "." for square in range(64))
    return replace(BoardObservation(frame, BASE_NS + frame * 1_000_000, status="ready", squares=values), **changes)


def initialize(session=None):
    session = session or ChessSession()
    board = chess.Board()
    seen = observation(board, orientation=session.orientation)
    proposal = session.propose(seen, now_ns=seen.captured_monotonic_ns)
    assert proposal.status == "proposed"
    assert session.confirm(proposal.proposal_id, now_ns=seen.captured_monotonic_ns + 1).status == "confirmed"
    return session, board


def advance(session, board, uci, frame):
    board.push_uci(uci)
    seen = observation(board, frame, orientation=session.orientation)
    proposed = session.propose(seen, now_ns=seen.captured_monotonic_ns + 1)
    assert proposed.status == "proposed", proposed.message
    assert proposed.uci == uci
    confirmed = session.confirm(proposed.proposal_id, now_ns=seen.captured_monotonic_ns + 2)
    assert confirmed.status == "confirmed"
    assert session.snapshot()["fen"] == board.fen()
    return confirmed


def test_initial_board_must_be_proposed_then_explicitly_confirmed():
    session = ChessSession()
    initial = session.snapshot()
    assert not initial["initialized"] and initial["fen"] is None and initial["confirmed_moves"] == []
    seen = observation(chess.Board())
    proposal = session.propose(seen, now_ns=seen.captured_monotonic_ns)
    assert proposal.status == "proposed" and proposal.proposal_id and proposal.uci is None
    assert not session.initialized and session.revision == 0
    assert session.snapshot()["pending"]["proposal_id"] == proposal.proposal_id
    assert "unconfirmed" in session.turn_description()
    confirmed = session.confirm(proposal.proposal_id, now_ns=seen.captured_monotonic_ns + 1)
    assert confirmed.status == "confirmed" and session.initialized and session.revision == 1
    assert session.pending is None
    assert session.snapshot()["fen"] == chess.STARTING_FEN
    assert session.turn_description() == "White to move."


@pytest.mark.parametrize("query", ["board", "turn", "last_move", "piece"])
def test_queries_do_not_invent_an_unconfirmed_position(query):
    session = ChessSession()
    assert "unconfirmed" in session.query(query, "e4")
    assert session.snapshot()["fen"] is None


def test_opening_move_changes_no_game_state_until_confirmation():
    session, board = initialize()
    before = session.snapshot()
    board.push_uci("e2e4")
    seen = observation(board, 2)
    proposed = session.propose(seen, now_ns=seen.captured_monotonic_ns)
    assert "White pawn from e2 to e4" in proposed.message
    assert session.snapshot()["fen"] == before["fen"]
    assert session.piece_at("e2") == "White pawn on e2."
    assert session.piece_at("e4") == "e4 is empty."
    assert session.snapshot()["confirmed_moves"] == []
    assert session.confirm(proposed.proposal_id, now_ns=seen.captured_monotonic_ns + 1).status == "confirmed"
    assert session.piece_at("e4") == "White pawn on e4."
    assert session.turn_description() == "Black to move."
    assert session.snapshot()["confirmed_moves"] == ["e2e4"]
    assert session.last_move_description() == "White pawn from e2 to e4."


def test_reject_and_pending_observation_cannot_overwrite_confirmed_history():
    session, board = initialize()
    before = session.snapshot()["fen"]
    board.push_uci("e2e4")
    seen = observation(board, 2)
    first = session.propose(seen, now_ns=seen.captured_monotonic_ns)
    second = session.propose(observation(chess.Board(), 3), now_ns=BASE_NS + 4_000_000)
    assert second.status == "pending" and second.proposal_id == first.proposal_id
    with pytest.raises(FrozenInstanceError):
        session.pending.uci = "d2d4"
    assert session.reject().status == "rejected"
    assert session.snapshot()["fen"] == before and session.snapshot()["confirmed_moves"] == []
    assert session.confirm(first.proposal_id, now_ns=seen.captured_monotonic_ns).status == "no_pending"


def test_ids_are_unique_and_old_confirmation_cannot_apply_to_a_new_proposal():
    session, board = initialize()
    board.push_uci("e2e4")
    first_seen = observation(board, 2)
    first = session.propose(first_seen, now_ns=first_seen.captured_monotonic_ns)
    session.reject()
    second_seen = observation(board, 3)
    second = session.propose(second_seen, now_ns=second_seen.captured_monotonic_ns)
    assert first.proposal_id != second.proposal_id
    assert session.confirm(first.proposal_id, now_ns=second_seen.captured_monotonic_ns).status == "invalid_confirmation"
    assert session.pending.proposal_id == second.proposal_id and session.revision == 1
    assert session.confirm(second.proposal_id, now_ns=second_seen.captured_monotonic_ns).status == "confirmed"
    assert session.confirm(second.proposal_id, now_ns=second_seen.captured_monotonic_ns).status == "no_pending"


def test_proposal_confirmation_expires_from_image_capture_not_response_time():
    session = ChessSession(max_age_s=1)
    seen = observation(chess.Board())
    proposed = session.propose(seen, now_ns=seen.captured_monotonic_ns + 900_000_000)
    assert proposed.status == "proposed"
    assert session.confirm(proposed.proposal_id, now_ns=seen.captured_monotonic_ns + 1_000_000_001).status == "stale"
    assert not session.initialized and session.pending is None


def test_changed_revision_invalidates_an_outstanding_proposal():
    session, board = initialize()
    board.push_uci("e2e4")
    seen = observation(board, 2)
    proposed = session.propose(seen, now_ns=seen.captured_monotonic_ns)
    # Simulate an external revision race; it must not commit a stale proposal.
    session.revision += 1
    assert session.confirm(proposed.proposal_id, now_ns=seen.captured_monotonic_ns).status == "stale"
    assert session.snapshot()["confirmed_moves"] == [] and session.pending is None


@pytest.mark.parametrize("changes", [
    {"status": "unclear"}, {"status": "failed"}, {"orientation": "black_near"},
    {"frame_id": 0}, {"frame_id": True}, {"squares": (".",) * 63},
    {"squares": ("?",) + (".",) * 63}, {"squares": ("X",) + (".",) * 63},
    {"squares": ([],) + (".",) * 63}, {"simulated": "false"},
])
def test_incomplete_or_malformed_observations_never_change_game(changes):
    session, board = initialize()
    before = session.snapshot()
    seen = observation(board, 2, **changes)
    result = session.propose(seen, now_ns=BASE_NS + 3_000_000)
    assert result.status == "invalid_observation"
    assert session.snapshot() == before


@pytest.mark.parametrize("changes", [
    {"frame_id": 1}, {"captured_monotonic_ns": BASE_NS + 1_000_000},
    {"captured_monotonic_ns": 0}, {"captured_monotonic_ns": True},
    {"captured_monotonic_ns": BASE_NS + 40_000_000_000},
])
def test_stale_reused_or_future_images_never_change_history(changes):
    session, board = initialize()
    before = session.snapshot()
    seen = observation(board, 2, **changes)
    assert session.propose(seen, now_ns=BASE_NS + 3_000_000).status == "stale"
    assert session.snapshot() == before


def test_old_complete_photo_is_stale_even_if_frame_id_is_new():
    session = ChessSession(max_age_s=1)
    seen = observation(chess.Board(), 50)
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns + 2_000_000_000).status == "stale"
    assert not session.initialized


def test_only_exact_standard_start_can_initialize_not_midgame_or_missing_piece():
    session = ChessSession()
    board = chess.Board()
    board.remove_piece_at(chess.A2)
    seen = observation(board)
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "inconsistent"
    board = chess.Board()
    board.push_uci("e2e4")
    seen = observation(board, 2)
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "inconsistent"
    assert not session.initialized


def test_no_change_advances_observation_metadata_but_not_position_or_revision():
    session, board = initialize()
    before = session.snapshot()
    seen = observation(board, 2)
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "no_change"
    after = session.snapshot()
    assert after["fen"] == before["fen"] and after["revision"] == before["revision"]
    assert after["accepted_frame_id"] == 2 and after["pending"] is None
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "stale"


def test_illegal_or_two_move_difference_cannot_be_inferred_as_one_move():
    session, board = initialize()
    before = session.snapshot()
    illegal = board.copy()
    illegal.remove_piece_at(chess.E2)
    illegal.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.WHITE))
    seen = observation(illegal, 2)
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "inconsistent"
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    seen = observation(board, 3)
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "inconsistent"
    assert session.snapshot() == before


def test_regular_capture_announces_both_piece_colors_and_squares():
    session, board = initialize()
    for frame, uci in enumerate(("e2e4", "d7d5", "e4d5"), 2):
        result = advance(session, board, uci, frame)
    assert "White pawn from e4 to d5, captures black pawn" in result.message
    assert session.piece_at("d5") == "White pawn on d5."


def test_castling_is_one_move_with_rook_movement_and_recorded_rights():
    session, board = initialize()
    for frame, uci in enumerate(("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"), 2):
        result = advance(session, board, uci, frame)
    assert "White king from e1 to g1, castling kingside; rook from h1 to f1" in result.message
    assert session.piece_at("f1") == "White rook on f1."
    assert session.snapshot()["fen"].split()[2] == "kq"


@pytest.mark.parametrize("sequence,announcement", [
    (("d2d4", "d7d5", "b1c3", "b8c6", "c1f4", "c8f5", "d1d2", "d8d7", "e1c1"),
     "White king from e1 to c1, castling queenside; rook from a1 to d1"),
    (("e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5", "d2d3", "e8g8"),
     "Black king from e8 to g8, castling kingside; rook from h8 to f8"),
])
def test_queenside_and_black_castling_announce_correct_rook_squares(sequence, announcement):
    session, board = initialize()
    for frame, uci in enumerate(sequence, 2):
        result = advance(session, board, uci, frame)
    assert announcement in result.message


def test_king_returning_to_start_does_not_restore_castling_rights_from_photo():
    session, board = initialize()
    sequence = ("g1f3", "a7a6", "e2e3", "a6a5", "f1e2", "b7b6", "e1f1", "b6b5", "f1e1", "c7c6")
    for frame, uci in enumerate(sequence, 2):
        advance(session, board, uci, frame)
    assert not board.has_kingside_castling_rights(chess.WHITE)
    imagined = board.copy()
    imagined.remove_piece_at(chess.E1)
    imagined.remove_piece_at(chess.H1)
    imagined.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    imagined.set_piece_at(chess.F1, chess.Piece(chess.ROOK, chess.WHITE))
    seen = observation(imagined, 20)
    before = session.snapshot()
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "inconsistent"
    assert session.snapshot() == before


def test_en_passant_uses_confirmed_last_move_and_announces_capture_square():
    session, board = initialize()
    for frame, uci in enumerate(("e2e4", "a7a6", "e4e5", "d7d5", "e5d6"), 2):
        result = advance(session, board, uci, frame)
    assert "White pawn from e5 to d6, captures black pawn en passant on d5" in result.message
    assert session.piece_at("d5") == "d5 is empty."


def test_expired_en_passant_cannot_be_inferred_from_piece_layout():
    session, board = initialize()
    for frame, uci in enumerate(("e2e4", "a7a6", "e4e5", "d7d5", "g1f3", "a6a5"), 2):
        advance(session, board, uci, frame)
    imagined = board.copy()
    imagined.remove_piece_at(chess.E5)
    imagined.remove_piece_at(chess.D5)
    imagined.set_piece_at(chess.D6, chess.Piece(chess.PAWN, chess.WHITE))
    seen = observation(imagined, 10)
    before = session.snapshot()
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "inconsistent"
    assert session.snapshot() == before


@pytest.mark.parametrize("promotion,piece_name", [("q", "queen"), ("r", "rook"), ("b", "bishop"), ("n", "knight")])
def test_promotion_is_disambiguated_by_observed_promoted_piece(promotion, piece_name):
    session, board = initialize()
    sequence = ("a2a4", "b7b5", "a4b5", "a7a6", "b5a6", "h7h6", "a6a7", "h6h5", "a7b8" + promotion)
    for frame, uci in enumerate(sequence, 2):
        result = advance(session, board, uci, frame)
    assert f"captures black knight, promotes to {piece_name}" in result.message
    assert session.piece_at("b8") == f"White {piece_name} on b8."


def test_check_and_checkmate_are_reported_without_suggesting_moves():
    session, board = initialize()
    for frame, uci in enumerate(("e2e4", "d7d5", "f1b5"), 2):
        result = advance(session, board, uci, frame)
    assert "Black is in check" in result.message and "in check" in session.turn_description()
    session, board = initialize()
    for frame, uci in enumerate(("f2f3", "e7e5", "g2g4", "d8h4"), 2):
        result = advance(session, board, uci, frame)
    assert "Checkmate. Black wins." in result.message
    assert session.turn_description() == "Checkmate. Black wins."
    before = session.snapshot()
    changed = list(observation(board, 6).squares)
    changed[chess.A2] = "."
    seen = observation(board, 6, squares=tuple(changed))
    assert session.propose(seen, now_ns=seen.captured_monotonic_ns).status == "game_over"
    assert session.snapshot() == before


def test_confirmed_board_queries_are_complete_bounded_and_repeatable():
    session, board = initialize(ChessSession(orientation="black_near"))
    text = session.board_description()
    assert len(text.split()) <= 60
    assert "White: king e1" in text and "Black: king e8" in text
    assert "a2, b2, c2, d2, e2, f2, g2, h2" in text
    assert session.query("repeat") == text
    assert session.query("piece", " E4 ") == "e4 is empty."
    assert "a1 through h8" in session.piece_at("e9")
    assert "No moves" in session.last_move_description()
    json.dumps(session.snapshot(), allow_nan=False)


def test_simulated_observation_remains_identified_in_proposal_and_snapshot():
    session = ChessSession()
    seen = observation(chess.Board(), simulated=True)
    proposed = session.propose(seen, now_ns=seen.captured_monotonic_ns)
    assert proposed.message.startswith("Simulated observation.")
    session.confirm(proposed.proposal_id, now_ns=seen.captured_monotonic_ns)
    assert session.snapshot()["simulated"] is True


def test_missing_optional_rules_dependency_is_friendly_and_sanitized(monkeypatch):
    original = builtins.__import__
    def no_chess(name, *args, **kwargs):
        if name == "chess":
            raise ImportError("PRIVATE_IMPORT_DETAIL")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_chess)
    with pytest.raises(ChessSessionError, match="optional chess") as caught:
        ChessSession()
    assert "PRIVATE_IMPORT_DETAIL" not in str(caught.value)


@pytest.mark.parametrize("age", [0, -1, True, float("nan"), float("inf"), 10 ** 1000])
def test_invalid_age_limits_fail_cleanly(age):
    with pytest.raises(ValueError, match="finite and positive"):
        ChessSession(max_age_s=age)
