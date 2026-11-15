"""Whole-phrase chess information commands; no spoken move is ever executed."""

from dataclasses import dataclass
import re
import unicodedata


_PHRASES = {
    "scan": ("check board", "check the board", "scan board", "scan the board", "what moved", "what has moved"),
    "confirm": ("confirm", "confirm move", "confirm the move", "confirm position", "confirm the position"),
    "reject": ("cancel move", "cancel the move", "reject move", "reject the move", "reject position", "cancel position"),
    "turn": ("whose turn", "whose turn is it", "who moves next", "what is the turn"),
    "last_move": ("last move", "what was the last move", "what was the last confirmed move", "tell me the last move"),
    "board": ("board", "describe board", "describe the board", "board information", "board status", "what is on the board"),
    "repeat": ("repeat", "repeat that", "say that again"),
    "stop": ("stop", "stop speaking", "stop talking", "be quiet"),
    "help": ("help", "chess help", "what can you do"),
}
_COMMANDS = {phrase: action for action, phrases in _PHRASES.items() for phrase in phrases}
_RANKS = {name: str(index) for index, name in enumerate(("one", "two", "three", "four", "five", "six", "seven", "eight"), 1)}
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\uff07": "'"})
_PIECE = re.compile(r"(?:what is on|what is at|what piece is on|which piece is on|piece on) ([a-h])\s*([1-8]|one|two|three|four|five|six|seven|eight)")


@dataclass(frozen=True)
class ChessCommand:
    action: str = "unknown"
    square: str | None = None

    @property
    def supported(self) -> bool:
        return self.action in _PHRASES or self.action == "piece"


def parse_chess_command(text: str) -> ChessCommand:
    """Parse one explicit request; negations, compounds, and move orders stay unknown."""
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        return ChessCommand()
    text = re.sub(r"\bwhat's\b", "what is", text.casefold().translate(_APOSTROPHES))
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    words = text.split()
    prefixes = set()
    while words and words[0] in ("vidi", "please") and words[0] not in prefixes:
        prefixes.add(words.pop(0))
    if words and words[-1] == "please":
        words.pop()
    text = " ".join(words)
    match = _PIECE.fullmatch(text)
    if match:
        file, rank = match.groups()
        return ChessCommand("piece", file + _RANKS.get(rank, rank))
    return ChessCommand(_COMMANDS.get(text, "unknown"))
