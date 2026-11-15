"""Optional local sherpa-onnx wake detection; never downloads or generates models.

The supported baseline is the English GigaSpeech 3.3M int8 KWS model. Keyword
tokens must come from its documented tokenizer, never a guessed token sequence.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Callable


class WakeWordError(RuntimeError):
    """A sanitized local model, dependency, or PCM-format failure."""


MODEL_FILES = {
    "encoder": "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "decoder": "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "joiner": "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "tokens": "tokens.txt",
}
MAX_PCM_BYTES = 16000 * 2 * 5


def tokenize_keyword(model_dir: Path, phrase: str = "HEY VIDI", *, tokenizer: Callable | None = None) -> str:
    """Explicitly produce one keyword-file line using sherpa's real BPE utility.

    No file is written. sherpa 2.0.1's utility needs sentencepiece AND pypinyin;
    its optional CLI also needs click. Runtime detection needs none of these
    tokenizer extras. This helper is never called by WakeDetector.
    """
    if not isinstance(phrase, str) or phrase.strip().upper() != "HEY VIDI":
        raise WakeWordError("The approved wake phrase is hey vidi.")
    directory = Path(model_dir)
    if not (directory / "tokens.txt").is_file() or not (directory / "bpe.model").is_file():
        raise WakeWordError("Local tokens.txt and bpe.model are required to encode the wake phrase.")
    try:
        if tokenizer is None:
            from sherpa_onnx import text2token
            tokenizer = text2token
        rows = tokenizer(["HEY VIDI"], tokens=str(directory / "tokens.txt"),
                         tokens_type="bpe", bpe_model=str(directory / "bpe.model"))
        if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or not rows[0]
                or any(not isinstance(token, str) or not token or any(c.isspace() for c in token) for token in rows[0])):
            raise ValueError
        return " ".join(rows[0]) + "\n"
    except ImportError:
        raise WakeWordError("Wake tokenization requires the approved sherpa-onnx, sentencepiece, and pypinyin packages.") from None
    except Exception:
        raise WakeWordError("The local tokenizer could not encode hey vidi with the selected model.") from None


class WakeDetector:
    """Lazy, CPU-only local keyword spotting from mono 16 kHz PCM16 bytes."""

    def __init__(self, model_dir: Path, keywords_path: Path, *, num_threads: int = 1,
                 threshold: float = .25, engine_factory: Callable | None = None) -> None:
        if type(num_threads) is not int or not 1 <= num_threads <= 4:
            raise WakeWordError("Wake detection needs one to four CPU threads.")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise WakeWordError("Wake threshold must be a finite number greater than zero and at most one.")
        self.model_dir, self.keywords_path = Path(model_dir), Path(keywords_path)
        self.num_threads, self.threshold = num_threads, threshold
        self._factory = engine_factory
        self._engine = self._stream = None
        self._lock = threading.RLock()
        self._closed = False
        self._close_requested = threading.Event()

    def _load(self) -> None:
        if self._engine is not None:
            return
        paths = {name: self.model_dir / filename for name, filename in MODEL_FILES.items()}
        if any(not path.is_file() for path in paths.values()) or not self.keywords_path.is_file():
            raise WakeWordError("Wake model files or the explicitly prepared keyword file are missing.")
        try:
            with self.keywords_path.open("rb") as source:
                data = source.read(16385)
            if len(data) > 16384 or len([line for line in data.decode("utf-8").splitlines() if line.strip()]) != 1:
                raise ValueError
            factory = self._factory
            if factory is None:
                from sherpa_onnx import KeywordSpotter
                factory = KeywordSpotter
            engine = factory(**{name: str(path) for name, path in paths.items()},
                             keywords_file=str(self.keywords_path), num_threads=self.num_threads,
                             sample_rate=16000, provider="cpu", keywords_threshold=float(self.threshold))
            stream = engine.create_stream()
            self._engine, self._stream = engine, stream
        except ImportError:
            raise WakeWordError("Local wake detection requires the approved optional sherpa-onnx package.") from None
        except Exception:
            raise WakeWordError("The local wake model or prepared keyword file could not be loaded.") from None

    def accept(self, pcm: bytes) -> bool:
        """Feed bounded PCM; reset immediately after a detected keyword."""
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > MAX_PCM_BYTES:
            raise WakeWordError("Wake input must be bounded mono 16 kHz PCM16 audio.")
        with self._lock:
            try:
                if self._closed or self._close_requested.is_set():
                    raise WakeWordError("The wake detector is closed.")
                if not pcm:
                    return False
                self._load()
                try:
                    import numpy as np
                    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.
                    self._stream.accept_waveform(16000, samples)
                    detected = False
                    while self._engine.is_ready(self._stream):
                        self._engine.decode_stream(self._stream)
                        if self._engine.get_result(self._stream):
                            self._engine.reset_stream(self._stream)
                            detected = True
                    return detected
                except Exception:
                    raise WakeWordError("The local wake detector could not process the microphone stream.") from None
            finally:
                self._release_requested()

    def prepare(self) -> None:
        """Validate and load before a caller opens the microphone hardware."""
        with self._lock:
            try:
                if self._closed or self._close_requested.is_set():
                    raise WakeWordError("The wake detector is closed.")
                self._load()
            finally:
                self._release_requested()

    def reset(self) -> None:
        with self._lock:
            if self._engine is not None and not self._closed:
                try:
                    self._engine.reset_stream(self._stream)
                except Exception:
                    raise WakeWordError("The local wake detector could not reset.") from None
                finally:
                    self._release_requested()

    def _release_requested(self) -> None:
        if self._close_requested.is_set():
            self._closed = True
            self._stream = self._engine = None

    def request_close(self) -> None:
        """Best-effort close without waiting for an active native inference lock."""
        self._close_requested.set()
        if self._lock.acquire(blocking=False):
            try:
                self._release_requested()
            finally:
                self._lock.release()

    def close(self) -> None:
        self._close_requested.set()
        with self._lock:
            self._release_requested()
