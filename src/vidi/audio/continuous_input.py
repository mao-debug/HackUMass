"""One persistent local microphone owner with bounded, private PCM buffers.

No recording is written to disk. Only capture_next returns a <=5-second WAV;
the caller alone decides whether to transcribe it. Pause closes the microphone
and clears every buffer. Wake detection continues during external TTS playback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import io
import math
import os
import subprocess
import threading
import time
import wave
from typing import Callable

from vidi.audio.speech_to_text import AudioClip, SpeechRecognitionError
from vidi.core.local_process import LocalProcessCancelled, LocalProcessError, _close_pipes, _stop_process, run_local_process

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
MAX_BUFFER_BYTES = BYTES_PER_SECOND * 5
CHUNK_BYTES = 640  # 20 ms; partial pipe reads are assembled into whole samples.


def _verify_input(device: str) -> None:
    if device != "pulse":
        return
    try:
        source = os.environ.get("PULSE_SOURCE", "").strip()
        if not source:
            result = run_local_process(["pactl", "get-default-source"], input_bytes=b"", timeout_s=2.)
            if result.returncode != 0 or len(result.stdout) > 1024:
                raise ValueError
            source = result.stdout.decode("utf-8").strip()
        if not source or source.endswith(".monitor") or any(c in source for c in "\x00\r\n"):
            raise ValueError
    except (LocalProcessError, UnicodeError, ValueError):
        raise SpeechRecognitionError("Select a verified headset microphone; an output monitor cannot be used.") from None


class AudioInput:
    """Persistent arecord input, with an independent bounded pipe-reader thread."""

    def __init__(self, device: str = "pulse", sample_rate: int = SAMPLE_RATE, *,
                 process_factory: Callable | None = None, source_validator: Callable | None = None) -> None:
        if not isinstance(device, str) or not device.strip() or any(c in device for c in "\x00\r\n"):
            raise SpeechRecognitionError("Choose a literal microphone input-device name.")
        if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
            raise SpeechRecognitionError("Continuous microphone input uses 16000 Hz mono PCM16.")
        self.device, self.sample_rate = device, sample_rate
        self._factory, self._validator = process_factory or subprocess.Popen, source_validator or _verify_input
        self._condition = threading.Condition()
        self._lifecycle = threading.Lock()
        self._chunks = deque()
        self._buffered = self.dropped_bytes = 0
        self._process = self._thread = None
        self._stop = threading.Event()
        self._stop.set()
        self._error = None

    @property
    def buffered_bytes(self) -> int:
        with self._condition:
            return self._buffered

    def start(self) -> None:
        with self._lifecycle:
            if self._process is not None:
                return
            self._validator(self.device)
            process = None
            thread = None
            try:
                process = self._factory(
                    ["arecord", "--quiet", "--device", self.device, "--file-type", "raw",
                     "--format", "S16_LE", "--rate", str(self.sample_rate), "--channels", "1"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=0, shell=False, start_new_session=os.name == "posix",
                )
                if process.stdout is None:
                    raise OSError
                with self._condition:
                    self._chunks.clear()
                    self._buffered = self.dropped_bytes = 0
                    self._error = None
                    self._stop = stop = threading.Event()
                    self._process = process
                self._thread = thread = threading.Thread(target=self._read_pipe, args=(process, stop), daemon=True,
                                                        name="vidi-microphone-pipe")
                thread.start()
            except BaseException as error:
                with self._condition:
                    self._process = None
                    self._thread = None
                    self._stop.set()
                    self._chunks.clear()
                    self._buffered = 0
                    self._condition.notify_all()
                if process is not None:
                    try:
                        _stop_process(process)
                    finally:
                        _close_pipes(process)
                        if thread is not None and thread.is_alive():
                            thread.join(.5)
                if isinstance(error, Exception):
                    raise SpeechRecognitionError("Continuous microphone capture could not start. Check arecord and the selected input.") from None
                raise

    def _read_pipe(self, process, stop) -> None:
        carry = b""
        try:
            while not stop.is_set():
                data = process.stdout.read(CHUNK_BYTES)
                if not data:
                    raise OSError
                if not isinstance(data, bytes) or len(data) > CHUNK_BYTES:
                    raise ValueError
                data = carry + data
                carry, data = data[len(data) // 2 * 2:], data[:len(data) // 2 * 2]
                if not data:
                    continue
                with self._condition:
                    if stop.is_set() or process is not self._process:
                        break
                    self._chunks.append(data)
                    self._buffered += len(data)
                    while self._buffered > MAX_BUFFER_BYTES:
                        dropped = self._chunks.popleft()
                        self._buffered -= len(dropped)
                        self.dropped_bytes += len(dropped)
                    self._condition.notify_all()
        except Exception:
            with self._condition:
                if not stop.is_set() and process is self._process:
                    self._error = "Continuous microphone input stopped. Check the headset connection."
                    self._chunks.clear()
                    self._buffered = 0
                self._condition.notify_all()

    def read(self, timeout_s: float = .1) -> bytes:
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 <= timeout_s <= 1:
            raise SpeechRecognitionError("Microphone read timeout must be between zero and one second.")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error:
                    raise SpeechRecognitionError(self._error)
                if self._stop.is_set():
                    raise SpeechRecognitionError("Continuous microphone input is not running.")
                if self._chunks:
                    data = self._chunks.popleft()
                    self._buffered -= len(data)
                    return data
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return b""
                self._condition.wait(remaining)

    def close(self) -> None:
        with self._lifecycle:
            with self._condition:
                process, self._process = self._process, None
                self._stop.set()
                self._chunks.clear()
                self._buffered = 0
                self._condition.notify_all()
            if process is not None:
                try:
                    _stop_process(process)
                finally:
                    _close_pipes(process)
                    if self._thread is not None and self._thread is not threading.current_thread():
                        self._thread.join(.5)


@dataclass
class _Capture:
    expected: int
    epoch: int
    data: bytearray = field(default_factory=bytearray, repr=False)


class ContinuousListener:
    """Keep local KWS active during playback; deliver one command clip at a time.

    on_wake runs on the listener thread and must return promptly: cancel TTS and
    enqueue a command event, never perform capture/transcription in the callback.
    Up to pre_roll_s of local audio is included INSIDE the <=5-second command
    budget, preserving words spoken before a streaming detector emits its hit.
    """

    def __init__(self, audio_input, detector, *, pre_roll_s: float = .3, cooldown_s: float = 1.) -> None:
        for value, limit in ((pre_roll_s, 1.), (cooldown_s, 5.)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                raise SpeechRecognitionError("Invalid continuous-listening timing settings.")
        self.audio_input, self.detector = audio_input, detector
        self.pre_roll_s, self.cooldown_s = pre_roll_s, cooldown_s
        self._condition = threading.Condition()
        self._lifecycle = threading.RLock()
        self._thread = None
        self._closed, self._paused = False, True
        self._error = None
        self._epoch = 0
        self._ring = bytearray()
        self._total_bytes = 0
        self._capture = None
        self._callback = None
        self._reset_needed = True
        self._last_wake_at = float("-inf")
        self._wake_index = None
        self._wake_id = self._used_wake_id = 0
        self._dropped_bytes = 0

    @property
    def error(self) -> str | None:
        with self._condition:
            return self._error

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    def start(self, on_wake: Callable[[], None]) -> None:
        if not callable(on_wake):
            raise SpeechRecognitionError("A wake callback is required.")
        with self._lifecycle:
            if self._closed:
                raise SpeechRecognitionError("The continuous listener is closed.")
            prepare = getattr(self.detector, "prepare", None)
            if prepare is not None:
                prepare()
            self._callback = on_wake
            self.resume()

    def _clear(self) -> None:
        self._epoch += 1
        self._ring.clear()
        if self._capture is not None:
            self._capture.data.clear()
        self._total_bytes = 0
        self._wake_index = None
        self._used_wake_id = self._wake_id
        self._reset_needed = True
        self._condition.notify_all()

    def pause(self) -> None:
        """Privacy pause: stop hardware and invalidate every buffered/captured byte."""
        with self._lifecycle:
            with self._condition:
                self._paused = True
                self._clear()
            self.audio_input.close()

    def resume(self) -> None:
        with self._lifecycle:
            with self._condition:
                if self._closed:
                    raise SpeechRecognitionError("The continuous listener is closed.")
                if self._callback is None:
                    raise SpeechRecognitionError("Start the continuous listener before resuming it.")
                if not self._paused:
                    return
            thread = None
            try:
                self.audio_input.start()
                with self._condition:
                    self._dropped_bytes = getattr(self.audio_input, "dropped_bytes", 0)
                    self._clear()
                    self._error = None
                    self._last_wake_at = float("-inf")
                    self._paused = False
                    self._condition.notify_all()
                if self._thread is None:
                    self._thread = thread = threading.Thread(target=self._run, daemon=True, name="vidi-local-wake")
                    thread.start()
            except BaseException as error:
                with self._condition:
                    self._paused = True
                    self._clear()
                    if thread is not None and not thread.is_alive():
                        self._thread = None
                    self._error = (str(error) if isinstance(error, SpeechRecognitionError) else
                                   "Continuous listening could not start. Check the microphone and local wake setup.")
                try:
                    self.audio_input.close()
                except Exception:
                    pass
                if isinstance(error, Exception):
                    raise SpeechRecognitionError(self._error) from None
                raise

    def _fail(self, error) -> None:
        from vidi.audio.wake_word import WakeWordError
        with self._condition:
            self._error = str(error) if isinstance(error, (SpeechRecognitionError, WakeWordError)) else "Continuous microphone or wake detection is unavailable."
        self.pause()

    def _run(self) -> None:
        try:
            self._listen()
        finally:
            # This existing owner can release native inference after it unwinds,
            # including when creating a separate cleanup thread was impossible.
            if self._closed:
                try:
                    self.detector.close()
                except Exception:
                    pass

    def _listen(self) -> None:
        while True:
            with self._condition:
                while self._paused and not self._closed:
                    self._condition.wait(.1)
                if self._closed:
                    return
                epoch = self._epoch
                reset, self._reset_needed = self._reset_needed, False
            try:
                if reset:
                    self.detector.reset()
                pcm = self.audio_input.read(timeout_s=.1)
                if getattr(self.audio_input, "dropped_bytes", 0) != self._dropped_bytes:
                    raise SpeechRecognitionError("Microphone processing fell behind. Pause and resume before trying again.")
                if not pcm:
                    continue
                if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > MAX_BUFFER_BYTES:
                    raise SpeechRecognitionError("Continuous input produced invalid PCM audio.")
                with self._condition:
                    if self._paused or epoch != self._epoch:
                        continue
                    self._ring.extend(pcm)
                    del self._ring[:max(0, len(self._ring) - MAX_BUFFER_BYTES)]
                    self._total_bytes += len(pcm)
                    capturing = self._capture is not None and self._capture.epoch == epoch
                    if capturing:
                        remaining = self._capture.expected - len(self._capture.data)
                        self._capture.data.extend(pcm[:remaining])
                        self._condition.notify_all()
                detected = self.detector.accept(pcm)
                with self._condition:
                    fire = (detected and not capturing and not self._paused and epoch == self._epoch and self._capture is None
                            and time.monotonic() - self._last_wake_at >= self.cooldown_s)
                    if fire:
                        self._last_wake_at = time.monotonic()
                        self._wake_id += 1
                        self._wake_index = max(0, self._total_bytes - int(self.pre_roll_s * BYTES_PER_SECOND) // 2 * 2)
                if fire:
                    self._callback()
            except Exception as error:
                with self._condition:
                    obsolete = self._paused or self._closed or epoch != self._epoch
                if not obsolete:
                    self._fail(error)

    def capture_next(self, seconds: float = 5., cancel_event: threading.Event | None = None) -> AudioClip:
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 5:
            raise SpeechRecognitionError("Command capture must be greater than zero and at most five seconds.")
        if threading.current_thread() is self._thread:
            raise SpeechRecognitionError("Command capture must run outside the wake callback.")
        expected = int(seconds * SAMPLE_RATE) * 2
        if expected < 2:
            raise SpeechRecognitionError("Command capture needs at least one PCM sample.")
        with self._condition:
            if self._closed or self._paused or self._thread is None:
                raise SpeechRecognitionError("The continuous microphone is paused or unavailable.")
            if self._capture is not None:
                raise SpeechRecognitionError("A command capture is already active.")
            capture = _Capture(expected, self._epoch)
            if self._wake_index is not None and self._wake_id > self._used_wake_id:
                start = self._wake_index - (self._total_bytes - len(self._ring))
                if start < 0:
                    raise SpeechRecognitionError("The wake-triggered command buffer expired.")
                capture.data.extend(self._ring[start:start + expected])
                self._used_wake_id = self._wake_id
            self._capture = capture
            deadline = time.monotonic() + seconds + 1.
            try:
                while len(capture.data) < expected:
                    if cancel_event is not None and cancel_event.is_set():
                        raise LocalProcessCancelled("Command capture was cancelled.")
                    if self._error:
                        raise SpeechRecognitionError(self._error)
                    if self._closed or self._paused or capture.epoch != self._epoch:
                        raise LocalProcessCancelled("Command capture was cancelled.")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SpeechRecognitionError("Continuous command capture did not finish in time.")
                    self._condition.wait(min(.05, remaining))
                if ((cancel_event is not None and cancel_event.is_set()) or self._closed or self._paused
                        or capture.epoch != self._epoch):
                    raise LocalProcessCancelled("Command capture was cancelled.")
                data = bytes(capture.data)
            finally:
                capture.data.clear()
                self._capture = None
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(data)
        with self._condition:
            if ((cancel_event is not None and cancel_event.is_set()) or self._closed or self._paused
                    or capture.epoch != self._epoch):
                raise LocalProcessCancelled("Command capture was cancelled.")
        return AudioClip(output.getvalue(), expected / BYTES_PER_SECOND)

    def close(self) -> None:
        with self._lifecycle:
            if self._closed:
                return
            with self._condition:
                self._closed = self._paused = True
                self._clear()
            try:
                self.audio_input.close()
            finally:
                if self._thread is not None and self._thread is not threading.current_thread():
                    self._thread.join(.5)
                if self._thread is None:
                    # Startup may have failed before a listener thread existed.
                    # Never wait for a native inference lock on the controller.
                    def release():
                        try:
                            self.detector.close()
                        except Exception:
                            pass

                    try:
                        threading.Thread(target=release, daemon=True, name="vidi-wake-close").start()
                    except Exception:
                        request_close = getattr(self.detector, "request_close", None)
                        if callable(request_close):
                            try:
                                request_close()
                            except Exception:
                                pass
