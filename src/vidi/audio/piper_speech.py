"""Optional offline Piper CLI synthesis using explicitly supplied local assets.

No Piper Python API is imported and no model is downloaded. Length scale is
passed directly to Piper; it is not a words-per-minute guarantee.
"""

from __future__ import annotations

import os
import base64
import io
import json
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import time
import threading
from typing import TYPE_CHECKING
import wave
import uuid

from vidi.audio.text_to_speech import _prepare_text, _validate_wav
from vidi.audio.wav_playback import play_wav
from vidi.core.local_process import LocalProcessCancelled, LocalProcessError, run_local_process
from vidi.core.types import SpeechResult

if TYPE_CHECKING:
    from vidi.core.configuration import SpeechSettings


class PiperSpeech:
    """Synthesize once and optionally play locally, with one shared deadline."""

    def __init__(self, settings: SpeechSettings) -> None:
        self.settings = settings

    def speak(self, text: str, *, output_path: Path | None = None,
              cancel_event: threading.Event | None = None) -> SpeechResult:
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
            if cancel_event is not None and cancel_event.is_set():
                raise LocalProcessCancelled("The speech request was cancelled.")
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise LocalProcessError("The speech deadline expired.")
            return budget

        if cancel_event is not None and cancel_event.is_set():
            return result("cancelled")

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
                **({"cancel_event": cancel_event} if cancel_event is not None else {}),
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
                play_wav(
                    temporary, timeout_s=remaining(), player=self.settings.player,
                    **({"cancel_event": cancel_event} if cancel_event is not None else {}),
                )
                remaining()
            return result("completed")
        except LocalProcessCancelled:
            return result("cancelled")
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


MAX_WARM_WAV_BYTES = 8 * 1024 * 1024
MAX_WARM_TEXT_CHARACTERS = 4096
_MAX_WARM_LINE = (MAX_WARM_WAV_BYTES * 4 // 3) + 4096

# This runs inside the explicitly configured speech environment. It imports
# neither vidi nor the camera environment's NumPy/OpenCV stack.
_WARM_WORKER_SOURCE = r"""
import base64, io, json, sys, wave
class BoundedWav(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > 8388608:
            raise ValueError('audio size')
        return super().write(data)
def emit(value):
    sys.stdout.write(json.dumps(value, separators=(',', ':')) + '\n')
    sys.stdout.flush()
try:
    from piper import PiperVoice, SynthesisConfig
    voice = PiperVoice.load(sys.argv[1])
    emit({'ready': True})
    while True:
        line = sys.stdin.buffer.readline(32769)
        if not line:
            break
        if len(line) > 32768:
            break
        request = json.loads(line)
        if request.get('close'):
            break
        token, text = request['id'], request['text']
        if not isinstance(text, str) or len(text) > 4096:
            break
        output = BoundedWav()
        with wave.open(output, 'wb') as audio:
            voice.synthesize_wav(text, audio, syn_config=SynthesisConfig(length_scale=request['length_scale']))
        data = output.getvalue()
        if len(data) > 8388608:
            emit({'id': token, 'error': 'size'})
        else:
            emit({'id': token, 'wav': base64.b64encode(data).decode('ascii')})
except BaseException:
    emit({'error': 'unavailable'})
"""


class _WarmWorker:
    """One persistent isolated voice process with bounded response framing."""

    def __init__(self, python: Path, model: Path):
        self.messages = queue.Queue(maxsize=2)
        self._closing = threading.Lock()
        self._closed = False
        self.process = subprocess.Popen(
            [str(python), "-u", "-c", _WARM_WORKER_SOURCE, str(model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            shell=False, start_new_session=os.name == "posix", close_fds=True,
        )

        def receive():
            try:
                while True:
                    line = self.process.stdout.readline(_MAX_WARM_LINE + 1)
                    if not line or len(line) > _MAX_WARM_LINE or not line.endswith(b"\n"):
                        self.messages.put_nowait(None)
                        return
                    value = json.loads(line)
                    self.messages.put_nowait(value)
            except (OSError, ValueError, queue.Full):
                try:
                    self.messages.put_nowait(None)
                except queue.Full:
                    pass

        self.reader = threading.Thread(target=receive, daemon=True, name="vidi-piper-replies")
        try:
            self.reader.start()
        except BaseException:
            self.close()
            raise

    def next(self, deadline, cancel):
        while True:
            if cancel.is_set():
                raise LocalProcessCancelled("Speech was cancelled.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LocalProcessError("Speech deadline expired.")
            try:
                value = self.messages.get(timeout=min(.05, remaining))
                if not isinstance(value, dict):
                    raise LocalProcessError("Speech worker stopped.")
                return value
            except queue.Empty:
                continue

    def synthesize(self, text, length_scale, deadline, cancel):
        token = uuid.uuid4().hex
        if cancel.is_set():
            raise LocalProcessCancelled("Speech was cancelled.")
        request = json.dumps({"id": token, "text": text, "length_scale": length_scale},
                             ensure_ascii=True, separators=(",", ":")).encode() + b"\n"
        if len(request) > 32768:
            raise LocalProcessError("Speech text exceeds the worker limit.")
        def send():
            try:
                self.process.stdin.write(request)
                self.process.stdin.flush()
            except (OSError, ValueError):
                try:
                    self.messages.put_nowait(None)
                except queue.Full:
                    pass
        # A pipe write must not hide cancellation/deadline checks if a worker
        # stops consuming input. The receiver wait remains bounded.
        threading.Thread(target=send, daemon=True, name="vidi-piper-request").start()
        reply = self.next(deadline, cancel)
        if set(reply) != {"id", "wav"} or reply.get("id") != token or not isinstance(reply["wav"], str):
            raise LocalProcessError("Speech worker returned invalid data.")
        if len(reply["wav"]) > ((MAX_WARM_WAV_BYTES + 2) // 3) * 4:
            raise LocalProcessError("Speech audio exceeds the worker limit.")
        data = base64.b64decode(reply["wav"], validate=True)
        if not data or len(data) > MAX_WARM_WAV_BYTES:
            raise LocalProcessError("Speech worker returned invalid audio.")
        return data

    def close(self):
        if not self._closing.acquire(blocking=False):
            return
        try:
            if self._closed:
                return
            self._closed = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    if os.name == "posix":
                        os.killpg(self.process.pid, sig)
                    elif sig == signal.SIGTERM:
                        self.process.terminate()
                    else:
                        self.process.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.process.wait(timeout=.2)
                except (subprocess.TimeoutExpired, OSError):
                    continue
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
        finally:
            self._closing.release()


class WarmPiperSpeech:
    """Load once per session; cancel playback and discard cancelled synthesis.

    A cancelled/failed inference destroys that worker before a later utterance
    can use it. Successful utterances reuse the loaded model. No assets are
    downloaded, and construction does not start the speech environment.
    """

    def __init__(self, settings: SpeechSettings, *, worker_factory=None):
        self.settings = settings
        self._worker_factory = worker_factory or _WarmWorker
        self._worker = None
        self._busy = threading.Lock()
        self._state = threading.Lock()
        self._active = None
        self._closed = False

    def stop(self):
        with self._state:
            if self._active is not None:
                self._active.set()

    def close(self):
        with self._state:
            self._closed = True
            if self._active is not None:
                self._active.set()
            worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()

    def speak(self, text: str, *, output_path=None, cancel_event=None, cancel=None) -> SpeechResult:
        started = time.monotonic()
        prepared, truncated = _prepare_text(text, self.settings.max_words) if isinstance(text, str) and text.strip() else ("", False)
        destination = Path(output_path) if output_path is not None else None
        model = self.settings.piper_model_path
        mode = "wav" if destination is not None else "playback"

        def result(status, error=None):
            return SpeechResult(status=status, mode=mode, backend="piper", text=prepared,
                                truncated=truncated, elapsed_s=max(0., time.monotonic() - started),
                                output_path=destination, error=error, voice=model.stem if model else None,
                                model=model.name if model else None)

        if not self._busy.acquire(blocking=False):
            return result("failed", "Speech is busy; no utterance was queued.")
        active = threading.Event()
        outer = cancel_event if cancel_event is not None else cancel

        class Cancel:
            def is_set(_self):
                return active.is_set() or (outer is not None and outer.is_set())

        cancelled = Cancel()
        temporary = worker = None
        inflight = False
        audio_ready = False
        deadline = started + self.settings.timeout_s

        def remaining():
            if cancelled.is_set():
                raise LocalProcessCancelled("Speech was cancelled.")
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise LocalProcessError("Speech deadline expired.")
            return budget

        try:
            with self._state:
                if self._closed:
                    return result("failed", "The speech session is closed.")
                self._active = active
            remaining()
            if not prepared or len(prepared) > MAX_WARM_TEXT_CHARACTERS:
                return result("failed", "Speech text is empty or exceeds the bounded utterance limit.")
            if model is None:
                return result("failed", "Piper requires a local ONNX voice model and its matching JSON configuration.")
            model = model.expanduser().resolve()
            companion = Path(str(model) + ".json")
            python = self.settings.piper_python_path or Path(sys.executable)
            python = python.expanduser().absolute()
            if (model.suffix.lower() != ".onnx" or not model.is_file() or not companion.is_file()
                    or not python.is_file() or not os.access(python, os.X_OK)):
                return result("failed", "The selected Piper runtime or voice assets are unavailable.")
            if destination is not None:
                if destination.suffix.lower() != ".wav" or destination.resolve() in (model, companion.resolve(), python.resolve()):
                    return result("failed", "Speech export requires a separate WAV destination.")
                destination.parent.mkdir(parents=True, exist_ok=True)
            with self._state:
                worker = self._worker
                if worker is None:
                    worker = self._worker_factory(python, model)
                    self._worker = worker
                    inflight = True
            if inflight:
                if worker.next(deadline, cancelled) != {"ready": True}:
                    raise LocalProcessError("Piper could not load the selected voice.")
            remaining()
            inflight = True
            data = worker.synthesize(prepared, self.settings.piper_length_scale, deadline, cancelled)
            if not isinstance(data, bytes) or not data or len(data) > MAX_WARM_WAV_BYTES:
                raise LocalProcessError("Piper returned invalid or oversized audio.")
            remaining()
            inflight = False
            with tempfile.NamedTemporaryFile(prefix=".vidi-piper-warm-", suffix=".wav",
                                             dir=destination.parent if destination else None, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            _validate_wav(temporary)
            remaining()
            audio_ready = True
            if destination is not None:
                temporary.replace(destination)
                temporary = None
            else:
                play_wav(temporary, timeout_s=remaining(), player=self.settings.player, cancel_event=cancelled)
                remaining()
            return result("completed")
        except LocalProcessCancelled:
            return result("cancelled")
        except (LocalProcessError, OSError, TypeError, ValueError, RuntimeError, wave.Error, EOFError):
            return result("cancelled" if cancelled.is_set() else "failed",
                          None if cancelled.is_set() else "Piper could not complete bounded synthesis or playback.")
        finally:
            if (inflight or (worker is not None and not audio_ready)) and worker is not None:
                with self._state:
                    if self._worker is worker:
                        self._worker = None
                worker.close()
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            with self._state:
                if self._active is active:
                    self._active = None
            self._busy.release()
