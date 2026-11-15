"""Recognize explicit English commands without executing model-produced text."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_PHRASES = {
    "describe": (
        "describe", "describe the scene", "describe my surroundings",
        "describe what you see", "what do you see", "what is in front of me",
        "what is in front of me now", "what is around me",
        "can you describe the scene", "can you describe my surroundings",
    ),
    "read_text": (
        "read text", "read this", "can you read this",
        "read this sign", "read the sign", "can you read this sign",
        "read this text", "read the text", "can you read this text",
        "read this label", "read the label", "can you read this label",
        "read this document", "read the document", "can you read this document",
        "read this page", "read the page", "can you read this page",
    ),
    "proximity": (
        "proximity", "check proximity", "check sensors", "check the sensors",
        "what is nearby", "what is near me", "is something nearby",
        "sensor status", "check obstacles",
    ),
    "stop": ("stop", "stop speaking", "stop talking", "be quiet", "silence"),
    "help": ("help", "help me", "what can you do"),
}
_COMMANDS = {phrase: action for action, phrases in _PHRASES.items() for phrase in phrases}
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\uff07": "'"})


@dataclass(frozen=True)
class Command:
    """One allowlisted action, or ``unknown``; text is the normalized utterance."""

    action: str = "unknown"
    text: str = ""

    @property
    def supported(self) -> bool:
        return self.action in _PHRASES


def parse_command(transcript: str) -> Command:
    """Match a whole phrase, never substrings or instructions embedded in prose.

    Case, punctuation, whitespace and "what's" are normalized. One leading
    "vidi" and one leading/trailing "please" are optional. Non-string, empty,
    and over-500-character inputs return unknown without retaining their text.
    Parsing opens no devices and invokes no capability or external service.
    """
    if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 500:
        return Command()
    text = transcript.casefold().translate(_APOSTROPHES)
    text = re.sub(r"\bwhat's\b", "what is", text)
    text = "".join(" " if unicodedata.category(character).startswith("P") else character
                   for character in text)
    words = text.split()
    # Permit "vidi please ..." and "please vidi ...", once per prefix.
    prefixes = set()
    while words and words[0] in ("vidi", "please") and words[0] not in prefixes:
        prefixes.add(words.pop(0))
    if words and words[-1] == "please":
        words.pop()
    text = " ".join(words)
    return Command(action=_COMMANDS.get(text, "unknown"), text=text)
