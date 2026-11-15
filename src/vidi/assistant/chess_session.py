"""Confirmed standard chess history with camera observations treated as proposals.

The optional rules package is imported only when a session is constructed.
This module contains no engine, move selection, camera, speech, or API client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import time
import uuid

from vidi.core.chess_types import BoardObservation, PIECE_SYMBOLS


class ChessSessionError(RuntimeError):
    """A local rules dependency is unavailable, without raw import diagnostics."""


@dataclass(frozen=True)
class Transition:
    status: str
    message: str
    proposal_id: str | None = None
    uci: str | None = None


@dataclass(frozen=True)
class PendingProposal:
    proposal_id: str
    revision: int
    frame_id: int
    captured_monotonic_ns: int
    uci: str | None
    message: str
    simulated: bool = False


_UNCONFIRMED = "Board position is unconfirmed. Scan the standard starting position and confirm it first."


class ChessSession:
    """Accept only a confirmed standard start and one uniquely legal move at a time."""

    def __init__(self, orientation: str = "white_near", max_age_s: float = 30):
        if orientation not in ("white_near", "black_near"):
            raise ValueError("Choose white_near or black_near board orientation.")
        try:
            valid_age = type(max_age_s) in (int, float) and math.isfinite(max_age_s) and max_age_s > 0
        except OverflowError:
            valid_age = False
        if not valid_age:
            raise ValueError("Chess observation age must be finite and positive.")
        try:
            import chess
        except ImportError:
            raise ChessSessionError("Chess rules are unavailable. Install the approved optional chess dependency.") from None
        self._chess = chess
        self._board = chess.Board()
        self.orientation = orientation
        self.max_age_s = max_age_s
        self.initialized = False
        self.revision = 0
        self._pending: PendingProposal | None = None
        self._accepted_frame_id = 0
        self._accepted_captured_ns = 0
        self._simulated = False
        self._last_move = None
        self._last_message = _UNCONFIRMED

    @property
    def pending(self) -> PendingProposal | None:
        return self._pending

    @property
    def last_message(self) -> str:
        return self._last_message

    def _reply(self, status, message, *, proposal=None):
        self._last_message = message
        return Transition(status, message, proposal.proposal_id if proposal else None,
                          proposal.uci if proposal else None)

    def _say(self, message):
        self._last_message = message
        return message

    def _layout(self, board):
        return tuple(piece.symbol() if (piece := board.piece_at(square)) else "." for square in range(64))

    def _fresh(self, captured_ns, now_ns):
        return (type(now_ns) is int and now_ns > 0 and type(captured_ns) is int and captured_ns > 0
                and 0 <= now_ns - captured_ns <= self.max_age_s * 1_000_000_000)

    def propose(self, observation: BoardObservation, now_ns: int | None = None) -> Transition:
        """Compare all 64 squares without changing the confirmed rules position."""
        if self._pending is not None:
            return self._reply("pending", "Confirm or reject the existing proposal before scanning again.", proposal=self._pending)
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if (not isinstance(observation, BoardObservation) or observation.status != "ready"
                or not isinstance(observation.squares, tuple) or len(observation.squares) != 64
                or any(type(symbol) is not str or symbol not in PIECE_SYMBOLS or symbol == "?" for symbol in observation.squares)
                or observation.orientation != self.orientation
                or type(observation.frame_id) is not int or observation.frame_id <= 0
                or type(observation.simulated) is not bool):
            return self._reply("invalid_observation", "Board observation is incomplete or has the wrong orientation. Confirmed history is unchanged.")
        if (not self._fresh(observation.captured_monotonic_ns, now_ns)
                or observation.frame_id <= self._accepted_frame_id
                or observation.captured_monotonic_ns <= self._accepted_captured_ns):
            return self._reply("stale", "Board observation is stale or already used. Capture a new board image.")
        if not self.initialized:
            if observation.squares != self._layout(self._board):
                return self._reply("inconsistent", "Only the standard starting layout can initialize this session. Arrange it and scan again.")
            move = None
            message = "Standard starting position detected. Awaiting confirmation before tracking moves."
        else:
            if observation.squares == self._layout(self._board):
                self._accept_observation(observation.frame_id, observation.captured_monotonic_ns, observation.simulated)
                return self._reply("no_change", "No change from the confirmed board position. " + self._turn())
            if self._board.is_game_over():
                return self._reply("game_over", "The confirmed game has ended. " + self._turn())
            matches = []
            for candidate in self._board.legal_moves:
                after = self._board.copy(stack=False)
                after.push(candidate)
                if self._layout(after) == observation.squares:
                    matches.append(candidate)
            if len(matches) != 1:
                status = "ambiguous" if matches else "inconsistent"
                return self._reply(status, "The observed board does not identify one legal move from confirmed history. Check the board and scan again.")
            move = matches[0]
            message = "Possible move: " + self._move_description(move)
        if observation.simulated:
            message = "Simulated observation. " + message
        pending = PendingProposal(uuid.uuid4().hex, self.revision, observation.frame_id,
                                  observation.captured_monotonic_ns, move.uci() if move else None,
                                  message, observation.simulated)
        self._pending = pending
        return self._reply("proposed", message, proposal=pending)

    def _accept_observation(self, frame_id, captured_ns, simulated):
        self._accepted_frame_id = frame_id
        self._accepted_captured_ns = captured_ns
        self._simulated = self._simulated or simulated

    def confirm(self, proposal_id: str, now_ns: int | None = None) -> Transition:
        """Commit exactly the still-current, unexpired proposal identified by its ID."""
        pending = self._pending
        if pending is None:
            return self._reply("no_pending", "There is no board proposal to confirm.")
        if not isinstance(proposal_id, str) or proposal_id != pending.proposal_id:
            return self._reply("invalid_confirmation", "That confirmation does not match the pending proposal.", proposal=pending)
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if (pending.revision != self.revision or not self._fresh(pending.captured_monotonic_ns, now_ns)
                or pending.frame_id <= self._accepted_frame_id
                or pending.captured_monotonic_ns <= self._accepted_captured_ns):
            self._pending = None
            return self._reply("stale", "The proposal expired or confirmed history changed. Capture a new board image.")
        if pending.uci is None:
            if self.initialized:
                self._pending = None
                return self._reply("invalid_confirmation", "Starting-position confirmation is no longer applicable.")
            self.initialized = True
            message = "Standard starting position confirmed. White to move."
        else:
            move = self._chess.Move.from_uci(pending.uci)
            if not self.initialized or move not in self._board.legal_moves:
                self._pending = None
                return self._reply("invalid_confirmation", "The proposed move no longer matches confirmed history.")
            announcement = self._move_description(move)
            self._board.push(move)
            self._last_move = announcement
            message = "Confirmed: " + announcement
        self.revision += 1
        self._accept_observation(pending.frame_id, pending.captured_monotonic_ns, pending.simulated)
        self._pending = None
        return self._reply("confirmed", message, proposal=pending)

    def reject(self) -> Transition:
        """Discard a hypothesis; never reset or otherwise modify confirmed history."""
        if self._pending is None:
            return self._reply("no_pending", "There is no board proposal to reject.")
        self._pending = None
        return self._reply("rejected", "Board proposal rejected. Confirmed history is unchanged.")

    def _move_description(self, move):
        chess = self._chess
        piece = self._board.piece_at(move.from_square)
        color = "White" if piece.color else "Black"
        text = f"{color} {chess.piece_name(piece.piece_type)} from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
        if self._board.is_castling(move):
            rank = "1" if piece.color else "8"
            side = "kingside" if self._board.is_kingside_castling(move) else "queenside"
            rook_from, rook_to = ("h", "f") if side == "kingside" else ("a", "d")
            text += f", castling {side}; rook from {rook_from}{rank} to {rook_to}{rank}"
        elif self._board.is_capture(move):
            en_passant = self._board.is_en_passant(move)
            captured_square = move.to_square + (-8 if piece.color else 8) if en_passant else move.to_square
            captured = self._board.piece_at(captured_square)
            text += f", captures {'white' if captured.color else 'black'} {chess.piece_name(captured.piece_type)}"
            if en_passant:
                text += f" en passant on {chess.square_name(captured_square)}"
        if move.promotion:
            text += ", promotes to " + chess.piece_name(move.promotion)
        after = self._board.copy(stack=True)
        after.push(move)
        text += "."
        if after.is_game_over():
            text += " " + self._turn(after)
        elif after.is_check():
            text += f" {'White' if after.turn else 'Black'} is in check."
        return text

    def _turn(self, board=None):
        board = self._board if board is None else board
        if board.is_checkmate():
            return f"Checkmate. {'Black' if board.turn else 'White'} wins."
        if board.is_stalemate():
            return "Game over: draw by stalemate."
        if board.is_insufficient_material():
            return "Game over: draw by insufficient material."
        if board.is_game_over():
            return "Game over: automatic draw under the recorded rules history."
        return f"{'White' if board.turn else 'Black'} to move." + (" The side to move is in check." if board.is_check() else "")

    def turn_description(self) -> str:
        return self._say(self._turn() if self.initialized else _UNCONFIRMED)

    def last_move_description(self) -> str:
        return self._say(_UNCONFIRMED if not self.initialized else self._last_move or "No moves have been confirmed.")

    def piece_at(self, square: str) -> str:
        if not self.initialized:
            return self._say(_UNCONFIRMED)
        if not isinstance(square, str) or not re.fullmatch("[a-h][1-8]", square.strip().lower()):
            return self._say("Ask for a square from a1 through h8.")
        square = square.strip().lower()
        piece = self._board.piece_at(self._chess.parse_square(square))
        return self._say(f"{'White' if piece.color else 'Black'} {self._chess.piece_name(piece.piece_type)} on {square}." if piece else f"{square} is empty.")

    def board_description(self) -> str:
        if not self.initialized:
            return self._say(_UNCONFIRMED)
        parts = ["Confirmed board."]
        for color, label in ((self._chess.WHITE, "White"), (self._chess.BLACK, "Black")):
            groups = []
            for kind in (self._chess.KING, self._chess.QUEEN, self._chess.ROOK, self._chess.BISHOP, self._chess.KNIGHT, self._chess.PAWN):
                names = [self._chess.square_name(square) for square in self._board.pieces(kind, color)]
                if names:
                    groups.append(self._chess.piece_name(kind) + ("s" if len(names) > 1 else "") + " " + ", ".join(names))
            parts.append(label + ": " + ("; ".join(groups) if groups else "no pieces") + ".")
        return self._say(" ".join(parts) + " " + self._turn())

    def query(self, action: str, square: str | None = None) -> str:
        if action == "repeat":
            return self._last_message
        if action == "piece":
            return self.piece_at(square)
        function = {"turn": self.turn_description, "last_move": self.last_move_description, "board": self.board_description}.get(action)
        return function() if function else self._say("Ask for the board, a square, whose turn, or the last confirmed move.")

    def snapshot(self) -> dict:
        """Serialize confirmed history and proposal metadata, never a rules object."""
        return {"initialized": self.initialized, "orientation": self.orientation, "revision": self.revision,
                "fen": self._board.fen() if self.initialized else None,
                "confirmed_moves": [move.uci() for move in self._board.move_stack] if self.initialized else [],
                "last_move": self._last_move, "pending": asdict(self._pending) if self._pending else None,
                "accepted_frame_id": self._accepted_frame_id, "accepted_captured_monotonic_ns": self._accepted_captured_ns,
                "simulated": self._simulated}
