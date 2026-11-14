"""Optional offline Piper CLI synthesis using explicitly supplied local assets.

No Piper Python API is imported and no model is downloaded. Length scale is
passed directly to Piper; it is not a words-per-minute guarantee.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
from typing import TYPE_CHECKING
import wave

from vidi.audio.text_to_speech import _prepare_text, _validate_wav
from vidi.audio.wav_playback import play_wav
from vidi.core.local_process import LocalProcessError, run_local_process
from vidi.core.types import SpeechResult

if TYPE_CHECKING:
    from vidi.core.configuration import SpeechSettings


class PiperSpeech:
    """Synthesize once and optionally play locally, with one shared deadline."""

    def __init__(self, settings: SpeechSettings) -> None:
        self.settings = settings

    def speak(self, text: str, *, output_path: Path | None = None) -> SpeechResult:
        started = time.monotonic()
        deadline = started + self.settings.timeout_s
        prepared, truncated = _prepare_text(text, self.settings.max_words)
        destination = Path(output_path) if output_path is not None else None
        mode = "wav" if destination is not None else "playback"
        temporary = None
        model = None

        def result(status: str, error: str | None = None) -> SpeechResult:
            return SpeechResult(
                status=status, mode=mode, backend="piper", text=prepared, truncated=truncated,
                elapsed_s=max(0.0, time.monotonic() - started), output_path=destination,
                error=error, voice=model.stem if model is not None else None,
                model=model.name if model is not None else None,
            )

        def remaining() -> float:
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise LocalProcessError("The speech deadline expired.")
            return budget

        try:
            if self.settings.piper_model_path is None:
                return result("failed", "Piper requires a local ONNX voice model and its matching JSON configuration.")
            model = self.settings.piper_model_path.expanduser().resolve()
            configuration = Path(str(model) + ".json")
            if model.suffix.lower() != ".onnx" or not model.is_file() or not configuration.is_file():
                return result("failed", "Piper requires an existing ONNX voice model and matching .onnx.json file.")
            python = self.settings.piper_python_path
            if python is not None:
                # A venv interpreter is often a symlink. Execute its configured
                # location so Python still discovers that environment's packages.
                python = python.expanduser().absolute()
                if not python.is_file() or not os.access(python, os.X_OK):
                    return result("failed", "The configured Piper Python executable is unavailable.")
            else:
                python = Path(sys.executable)
            if destination is not None:
                if destination.resolve() in (model, configuration.resolve(), python.resolve()):
                    return result("failed", "Speech output must not overwrite Piper voice assets or its Python runtime.")
                if destination.suffix.lower() != ".wav":
                    return result("failed", "Piper speech export requires a .wav destination.")
                destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".vidi-piper-", suffix=".wav", dir=destination.parent if destination else None,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
            completed = run_local_process(
                [str(python), "-m", "piper", "--model", str(model), "--output-file", str(temporary),
                 "--length-scale", str(self.settings.piper_length_scale)],
                input_bytes=prepared.encode("utf-8"), timeout_s=remaining(),
            )
            if completed.returncode != 0:
                return result("failed", "Piper synthesis failed. Check the local runtime and voice model.")
            remaining()
            _validate_wav(temporary)
            remaining()
            if destination is not None:
                temporary.replace(destination)
                temporary = None
            else:
                play_wav(temporary, timeout_s=remaining(), player=self.settings.player)
                remaining()
            return result("completed")
        except LocalProcessError:
            return result("failed", "Piper synthesis or playback could not complete. Check local setup and timeout.")
        except (OSError, UnicodeError, wave.Error, EOFError, ValueError, RuntimeError):
            return result("failed", "Piper could not create valid PCM WAV audio or export the requested file.")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
