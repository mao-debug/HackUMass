"""One bounded playback of a caller-validated WAV through an installed player."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import sys

from vidi.core.local_process import LocalProcessError, run_local_process


_ALLOWED_PLAYERS = {"paplay", "pw-play", "afplay", "aplay"}


def play_wav(path: Path, *, timeout_s: float, player: str | None = None) -> str:
    """Play once, returning the player name; never retry or change audio routes.

    Automatic selection tries paplay then pw-play on Linux, or afplay on Mac.
    ALSA's aplay requires explicit selection. Content validation belongs to the
    synthesizer; this boundary requires only an existing regular input file.
    """
    try:
        valid_timeout = type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s > 0
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise LocalProcessError("WAV playback requires a finite positive timeout.")
    if player is not None and (not isinstance(player, str) or player not in _ALLOWED_PLAYERS):
        raise LocalProcessError("Choose an installed paplay, pw-play, afplay, or explicit aplay player.")
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise LocalProcessError("WAV playback requires an existing regular file.")
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        if isinstance(error, LocalProcessError):
            raise
        raise LocalProcessError("WAV playback requires an existing regular file.") from None

    if player is not None:
        candidates = (player,)
    elif sys.platform == "darwin":
        candidates = ("afplay",)
    elif sys.platform.startswith("linux"):
        candidates = ("paplay", "pw-play")
    else:
        raise LocalProcessError("No automatic WAV player is supported on this platform; select an installed player.")
    selected = executable = None
    try:
        for candidate in candidates:
            executable = shutil.which(candidate)
            if executable:
                selected = candidate
                break
    except OSError:
        raise LocalProcessError("The installed WAV player could not be located.") from None
    if selected is None:
        raise LocalProcessError("No selected WAV player is installed. Check the available local audio tools.")
    completed = run_local_process([executable, str(resolved)], input_bytes=b"", timeout_s=timeout_s)
    if completed.returncode != 0:
        raise LocalProcessError("WAV playback failed. Check the selected player and audio output device.")
    return selected
