"""One-shot local speech or explicit WAV export, without a playback queue.

Recognized text is input data, never a command-line argument or markup. The
caller retains the complete OCR result; only this speech copy is bounded and
has engine control delimiters neutralized.
"""

from __future__ import annotations

import re
import sys
import tempfile
import time
from pathlib import Path
import wave

from vidi.core.configuration import SpeechSettings
from vidi.core.local_process import LocalProcessError, run_local_process
from vidi.core.types import SpeechResult


NO_TEXT_FEEDBACK = "No readable text found."
TRUNCATION_FEEDBACK = "More text is available."


def _prepare_text(text: str, max_words: int) -> tuple[str, bool]:
    """Bound recognized words while preserving printable text and punctuation."""
    cleaned = "".join(character if character.isprintable() else " " for character in text).strip()
    words = list(re.finditer(r"\S+", cleaned))
    if not words:
        return NO_TEXT_FEEDBACK, False
    truncated = len(words) > max_words
    if truncated:
        cleaned = cleaned[: words[max_words - 1].end()] + " " + TRUNCATION_FEEDBACK
    # Native say supports [[...]] speech commands. Also neutralize them for
    # eSpeak so OCR is treated consistently as text, including longer runs.
    cleaned = re.sub(r"\[(?=\[)", "[ ", cleaned)
    cleaned = re.sub(r"\](?=\])", "] ", cleaned)
    return cleaned, truncated


def _validate_wav(path: Path) -> None:
    """Require a nonempty complete PCM wave, not just a successful exit code."""
    with wave.open(str(path), "rb") as audio:
        frame_bytes = audio.getnchannels() * audio.getsampwidth()
        remaining = audio.getnframes()
        if audio.getcomptype() != "NONE" or frame_bytes <= 0 or audio.getframerate() <= 0 or remaining <= 0:
            raise ValueError("Speech engine did not create nonempty PCM WAV audio.")
        while remaining:
            data = audio.readframes(min(remaining, 4096))
            if not data or len(data) % frame_bytes:
                raise ValueError("Speech engine created incomplete WAV audio.")
            remaining -= len(data) // frame_bytes


class LocalSpeech:
    """Speak once after explicit user action, or export without playing audio."""

    def __init__(self, settings: SpeechSettings, backend: str | None = None) -> None:
        self.settings = settings
        self.backend = backend

    def speak(self, text: str, *, output_path: Path | None = None) -> SpeechResult:
        """Return an observable result; Ctrl+C propagates after local cleanup.

        File export uses the selected WAV-capable engine and validates a sibling temporary
        file before replacing the destination. A failed or cancelled export
        preserves any existing destination. No retry or fallback is attempted.
        """
        if self.backend == "piper":
            from vidi.audio.piper_speech import PiperSpeech

            return PiperSpeech(self.settings).speak(text, output_path=output_path)
        started = time.monotonic()
        mode = "wav" if output_path is not None else "playback"
        backend = self.backend
        if backend is None:
            if mode == "wav" or sys.platform.startswith("linux"):
                backend = "espeak_ng"
            elif sys.platform == "darwin":
                backend = "say"
        prepared, truncated = _prepare_text(text, self.settings.max_words)
        destination = Path(output_path) if output_path is not None else None
        temporary: Path | None = None

        def result(status: str, error: str | None = None) -> SpeechResult:
            return SpeechResult(
                status=status, mode=mode, backend=backend, text=prepared,
                truncated=truncated, elapsed_s=max(0.0, time.monotonic() - started),
                output_path=destination, error=error,
                voice=self.settings.voice or {"say": "Samantha", "espeak_ng": "en"}.get(backend),
            )

        if backend not in ("espeak_ng", "say"):
            return result("failed", "No supported local speech backend is configured for this platform.")
        if mode == "wav" and backend == "say":
            return result("failed", "WAV export requires espeak_ng or piper; the say backend supports playback only.")

        try:
            if backend == "say":
                argv = ["say", "-v", self.settings.voice or "Samantha", "-r", str(self.settings.speed_wpm), "-f", "-"]
            else:
                argv = ["espeak-ng", "-v", self.settings.voice or "en", "-s", str(self.settings.speed_wpm), "--stdin"]
                if destination is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        prefix=f".{destination.name}.", suffix=".wav", dir=destination.parent, delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                    argv.extend(("-w", str(temporary)))

            completed = run_local_process(
                argv, input_bytes=prepared.encode("utf-8"), timeout_s=self.settings.timeout_s
            )
            if completed.returncode != 0:
                return result(
                    "failed",
                    f"Local {backend} speech exited with status {completed.returncode}. "
                    "Check the installed engine, selected voice, and output device.",
                )
            if temporary is not None:
                _validate_wav(temporary)
                temporary.replace(destination)
                temporary = None
            return result("completed")
        except LocalProcessError as error:
            return result("failed", f"Local speech could not finish: {error}")
        except (OSError, UnicodeError, wave.Error, EOFError, ValueError):
            return result("failed", "Cannot export speech: check the output path and valid nonempty PCM WAV audio.")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    # Preserve the original failure or interruption if cleanup
                    # is denied by a changed filesystem permission.
                    pass
