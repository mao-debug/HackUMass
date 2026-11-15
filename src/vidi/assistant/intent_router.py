"""Shared whole-utterance intent routing for headless normal/chess modes."""

from dataclasses import dataclass
import re

from vidi.assistant.chess_commands import parse_chess_command
from vidi.assistant.command_handler import parse_command


@dataclass(frozen=True)
class Intent:
    action: str = "unknown"
    square: str | None = None


def route_intent(text, mode="normal"):
    if not isinstance(text, str) or not 0 < len(text) <= 500:
        return Intent()
    normalized = " ".join(text.casefold().strip().split()).strip(".!?")
    normalized = re.sub(r"^(?:hey\s+)?vidi(?:[\s,.:;!?]+|$)", "", normalized)
    if normalized.startswith("please "):
        normalized = normalized[7:]
    switches = {
        "chess mode": "mode_chess", "switch to chess mode": "mode_chess",
        "normal mode": "mode_normal", "switch to normal mode": "mode_normal",
        "exit chess mode": "mode_normal", "stop listening": "privacy_pause",
        "capture an image": "capture", "capture image": "capture",
        "take a photo": "capture", "take a picture": "capture",
        "forget recent context": "forget", "forget that": "forget",
        "retry": "retry", "try again": "retry",
    }
    if normalized in switches:
        return Intent(switches[normalized])
    if mode == "chess":
        command = parse_chess_command(normalized)
        return Intent(command.action, command.square)
    aliases = {
        "what is directly in front of me": "describe",
        "what's directly in front of me": "describe",
        "describe the entire area around me": "describe",
        "read the text in front of me": "read_text",
        "read the text infront of me": "read_text",
        "repeat": "repeat", "repeat that": "repeat",
    }
    return Intent(aliases.get(normalized, parse_command(normalized).action))

