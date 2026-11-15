"""Explicit speech selection for a session, without automatic provider fallback."""

from __future__ import annotations

import re
import threading

from vidi.core.types import SpeechResult


SpeechOutcome = SpeechResult
_RANKS = dict(zip("12345678", ("one", "two", "three", "four", "five", "six", "seven", "eight")))


def render_chess_speech(text: str) -> str:
    """Render isolated chess coordinates for speech without changing stored moves."""
    return re.sub(r"\b([a-h])([1-8])\b",
                  lambda match: match[1].upper() + " " + _RANKS[match[2]], text, flags=re.IGNORECASE)


class SpeechSession:
    """One selected engine with stop/close and no queued utterances.

    Construction is inert. A Gemini engine additionally needs cloud_enabled
    from explicit runtime enablement; a key or YAML backend alone is insufficient.
    """

    def __init__(self, settings, backend=None, *, cloud_enabled=False):
        if backend not in (None, "say", "espeak_ng", "piper", "gemini"):
            raise ValueError("Choose a configured local or Gemini speech backend.")
        self.settings, self.backend = settings, backend
        self.cloud_enabled = cloud_enabled is True
        self._engine = None
        self._busy = threading.Lock()
        self._state = threading.Lock()
        self._active = None
        self._closed = False

    def _get_engine(self):
        if self._engine is None:
            if self.backend == "piper":
                from vidi.audio.piper_speech import WarmPiperSpeech
                self._engine = WarmPiperSpeech(self.settings)
            elif self.backend == "gemini":
                from vidi.audio.gemini_speech import GeminiSpeech
                self._engine = GeminiSpeech(self.settings, cloud_enabled=self.cloud_enabled)
            else:
                from vidi.audio.text_to_speech import LocalSpeech
                self._engine = LocalSpeech(self.settings, self.backend)
        return self._engine

    def speak(self, text, *, cancel=None, cancel_event=None, output_path=None):
        if not self._busy.acquire(blocking=False):
            return SpeechResult(status="failed", backend=self.backend, error="Speech is busy; no utterance was queued.")
        active = threading.Event()
        outer = cancel_event if cancel_event is not None else cancel

        class Cancel:
            def is_set(_self):
                return active.is_set() or (outer is not None and outer.is_set())

        try:
            with self._state:
                if self._closed:
                    return SpeechResult(status="failed", backend=self.backend, error="The speech session is closed.")
                self._active = active
                engine = self._get_engine()
            if Cancel().is_set():
                return SpeechResult(status="cancelled", backend=self.backend)
            return engine.speak(text, output_path=output_path, cancel_event=Cancel())
        finally:
            with self._state:
                if self._active is active:
                    self._active = None
            self._busy.release()

    def stop(self):
        with self._state:
            if self._active is not None:
                self._active.set()
            engine = self._engine
        if engine is not None and hasattr(engine, "stop"):
            engine.stop()

    def close(self):
        with self._state:
            self._closed = True
        self.stop()
        engine = self._engine
        if engine is not None and hasattr(engine, "close"):
            engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def create_speech(settings, backend=None, *, cloud_enabled=False) -> SpeechSession:
    return SpeechSession(settings, backend, cloud_enabled=cloud_enabled)
