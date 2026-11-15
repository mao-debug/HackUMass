"""Explicit Gemini text-to-speech with bounded streaming and local cancellation.

No client, key, or playback device is accessed during construction. Selecting
this backend still requires the caller's explicit cloud-enabled session.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
import io
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave

from vidi.audio.text_to_speech import _prepare_text
from vidi.audio.wav_playback import play_wav
from vidi.core.local_process import LocalProcessCancelled, LocalProcessError
from vidi.core.types import SpeechResult


SAMPLE_RATE = 24000
MAX_PCM_BYTES = SAMPLE_RATE * 2 * 120
MAX_TEXT_CHARACTERS = 4096
MAX_EVENTS = 2048


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _stop_player(process):
    """Stop the dedicated playback process group without waiting on audio I/O."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, OSError):
        pass


class _PCMPlayer:
    """Bounded raw PCM queue for the existing PulseAudio playback command."""

    def __init__(self, executable, deadline, cancel):
        self.deadline, self.cancel = deadline, cancel
        self.chunks = queue.Queue(maxsize=8)
        self.done = threading.Event()
        self.stopped = threading.Event()
        self.failed = False
        self.process = subprocess.Popen(
            [executable, "--raw", "--format=s16le", "--rate=24000", "--channels=1"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            shell=False, start_new_session=os.name == "posix", close_fds=True,
        )

        def write():
            try:
                while True:
                    self.check()
                    try:
                        chunk = self.chunks.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if chunk is None:
                        self.process.stdin.close()
                        while self.process.poll() is None:
                            self.check()
                            time.sleep(.02)
                        if self.process.returncode != 0:
                            self.failed = True
                        return
                    self.process.stdin.write(chunk)
                    self.process.stdin.flush()
            except (OSError, ValueError, LocalProcessError):
                self.failed = True
                _stop_player(self.process)
            finally:
                try:
                    self.process.stdin.close()
                except (OSError, ValueError):
                    pass
                self.done.set()

        self.writer = threading.Thread(target=write, daemon=True, name="vidi-gemini-pcm")
        try:
            self.writer.start()
        except BaseException:
            self.close()
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            raise

    def check(self):
        if self.stopped.is_set() or self.cancel.is_set():
            raise LocalProcessCancelled("Speech was cancelled.")
        if time.monotonic() >= self.deadline:
            raise LocalProcessError("Speech playback deadline expired.")

    def _put(self, data):
        while True:
            self.check()
            if self.done.is_set():
                raise LocalProcessError("Speech playback stopped.")
            try:
                self.chunks.put(data, timeout=.05)
                return
            except queue.Full:
                continue

    def write(self, data):
        for start in range(0, len(data), 32768):
            self._put(data[start:start + 32768])

    def finish(self):
        self._put(None)
        while not self.done.wait(.02):
            self.check()
        self.check()
        if self.failed:
            raise LocalProcessError("Speech playback failed.")

    def stop(self):
        self.stopped.set()
        _stop_player(self.process)

    def close(self):
        self.stop()
        try:
            self.process.wait(timeout=.2)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _make_player(settings, deadline, cancel):
    # Other configured players and Mac use a validated complete WAV. No audio
    # server is installed or routing changed to obtain streaming playback.
    if sys.platform.startswith("linux") and settings.player in (None, "paplay"):
        executable = shutil.which("paplay")
        if executable:
            return _PCMPlayer(executable, deadline, cancel)
    return None


class GeminiSpeech:
    """One selected cloud utterance; no retry, stored history, or provider fallback."""

    def __init__(self, settings, *, cloud_enabled=False, client_factory=None, player_factory=None):
        self.settings = settings
        self.cloud_enabled = cloud_enabled is True
        self._client_factory = client_factory
        self._player_factory = player_factory or _make_player
        self._busy = threading.Lock()
        self._state = threading.Lock()
        self._active = self._player = self._stream = self._client = None
        self._closed = False
        self._cleanup_threads = {}
        self._cleanup_seen = set()
        self._cleanup_failed = False

    def _release_transports(self, resources, *, wait_s=0.):
        """Bound cleanup and reject later utterances while it remains pending."""
        for resource in resources:
            if resource is None:
                continue
            token = id(resource)
            with self._state:
                if token in self._cleanup_seen:
                    continue
                self._cleanup_seen.add(token)

                def release(selected=resource, selected_token=token):
                    try:
                        selected.close()
                    except Exception:
                        pass
                    finally:
                        with self._state:
                            self._cleanup_threads.pop(selected_token, None)

                worker = threading.Thread(target=release, daemon=True, name="vidi-tts-release")
                self._cleanup_threads[token] = worker
            try:
                worker.start()
            except RuntimeError:
                with self._state:
                    self._cleanup_threads.pop(token, None)
                    self._cleanup_failed = True
        if wait_s:
            deadline = time.monotonic() + wait_s
            with self._state:
                workers = tuple(self._cleanup_threads.values())
            for worker in workers:
                try:
                    worker.join(timeout=max(0., deadline - time.monotonic()))
                except RuntimeError:
                    # A simultaneous stop can register a worker immediately
                    # before starting it. Its tracked cleanup still gates reuse.
                    pass

    def stop(self):
        with self._state:
            already_stopped = self._active is None or self._active.is_set()
            if self._active is not None:
                self._active.set()
            player = self._player
            resources = (self._stream, self._client)
        if player is not None:
            player.stop()
        if not already_stopped and any(resource is not None for resource in resources):
            # Closing a streaming transport is best effort. It must never
            # block the controller's local stop or claim server cancellation.
            self._release_transports(resources)

    def close(self):
        with self._state:
            self._closed = True
        self.stop()

    def speak(self, text, *, output_path=None, cancel_event=None, cancel=None):
        started = time.monotonic()
        prepared, truncated = _prepare_text(text, self.settings.max_words) if isinstance(text, str) and text.strip() else ("", False)
        # Literal source text must not become Gemini vocal-event tags.
        prepared = prepared.replace("<", " less than ").replace(">", " greater than ")
        destination = Path(output_path) if output_path is not None else None
        model = self.settings.gemini_model
        voice = self.settings.gemini_voice

        def result(status, error=None):
            return SpeechResult(status=status, mode="wav" if destination else "playback", backend="gemini",
                                text=prepared, truncated=truncated, elapsed_s=max(0., time.monotonic() - started),
                                output_path=destination, error=error, model=model, voice=voice)

        if not self.cloud_enabled:
            return result("failed", "Gemini speech requires explicit cloud-speech enablement.")
        if not self._busy.acquire(blocking=False):
            return result("failed", "Speech is busy; no utterance was queued.")
        active = threading.Event()
        outer = cancel_event if cancel_event is not None else cancel

        class Cancel:
            def is_set(_self):
                return active.is_set() or (outer is not None and outer.is_set())

        cancelled = Cancel()
        timeout = min(float(self.settings.timeout_s), 120.)
        deadline = started + timeout
        client = stream = player = temporary = None

        def check():
            if cancelled.is_set():
                raise LocalProcessCancelled("Speech was cancelled.")
            if time.monotonic() >= deadline:
                raise LocalProcessError("Speech deadline expired.")

        try:
            with self._state:
                if self._closed:
                    return result("failed", "The speech session is closed.")
                if self._cleanup_threads or self._cleanup_failed:
                    return result("failed", "Prior speech transport cleanup is unfinished; no utterance was queued.")
                self._cleanup_seen.clear()
                self._active = active
            check()
            if not prepared or len(prepared) > MAX_TEXT_CHARACTERS:
                return result("failed", "Speech text is empty or exceeds the bounded utterance limit.")
            if destination is not None and destination.suffix.lower() != ".wav":
                return result("failed", "Speech export requires a WAV destination.")
            key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not key:
                return result("failed", "A Gemini key is required for explicitly selected cloud speech.")
            if key.casefold() in prepared.casefold():
                prepared = ""
                return result("failed", "Speech text contains credential data and was not sent.")
            from vidi.ai.scene_description import _default_client_factory

            factory = self._client_factory or _default_client_factory
            check()
            client = factory(api_key=key, http_options={
                "timeout": math.ceil(timeout * 1000), "retry_options": {"attempts": 0},
            })
            with self._state:
                self._client = client
            check()
            annotations = [{"type": "speech_metadata", "style": self.settings.gemini_style}] if self.settings.gemini_style else []
            stream = client.interactions.create(
                model=model, input=[{"type": "user_input", "content": [
                    {"type": "text", "text": prepared, "annotations": annotations},
                ]}],
                response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": SAMPLE_RATE},
                generation_config={"speech_config": [{"voice": voice}],
                                   "max_output_tokens": min(3000, max(1, math.ceil(timeout * 25)))},
                tools=[], store=False, background=False, stream=True,
                automatic_continuation=False, timeout=timeout,
            )
            with self._state:
                self._stream = stream
            check()
            pcm, completed, carry = bytearray(), False, b""
            playback_decided = False
            for count, event in enumerate(stream, 1):
                check()
                if count > MAX_EVENTS:
                    raise ValueError
                kind = _field(event, "event_type")
                if kind == "interaction.completed":
                    if _field(_field(event, "interaction"), "status") != "completed":
                        raise ValueError
                    completed = True
                    break
                if kind in ("error", "interaction.failed", "interaction.incomplete"):
                    raise ValueError
                if kind != "step.delta":
                    continue
                delta = _field(event, "delta")
                if _field(delta, "type") != "audio" or _field(delta, "uri") is not None:
                    raise ValueError
                rate, channels = _field(delta, "sample_rate"), _field(delta, "channels")
                if (_field(delta, "mime_type") not in (None, "audio/l16")
                        or (rate is not None and (type(rate) is not int or rate != SAMPLE_RATE))
                        or (channels is not None and (type(channels) is not int or channels != 1))):
                    raise ValueError
                encoded = _field(delta, "data")
                if encoded is None:
                    continue
                if not isinstance(encoded, str) or len(encoded) > ((MAX_PCM_BYTES + 2) // 3) * 4:
                    raise ValueError
                data = base64.b64decode(encoded, validate=True)
                if len(pcm) + len(data) > MAX_PCM_BYTES:
                    raise ValueError
                pcm.extend(data)
                data = carry + data
                usable = len(data) - len(data) % 2
                carry = data[usable:]
                if usable and destination is None:
                    if not playback_decided:
                        player = self._player_factory(self.settings, deadline, cancelled)
                        playback_decided = True
                        with self._state:
                            self._player = player
                        check()
                    if player is not None:
                        player.write(data[:usable])
            check()
            if not completed or not pcm or carry:
                raise ValueError
            if player is not None:
                player.finish()
                check()
            else:
                if destination is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix=".vidi-gemini-tts-", suffix=".wav",
                                                 dir=destination.parent if destination else None, delete=False) as file:
                    temporary = Path(file.name)
                    with wave.open(file, "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(SAMPLE_RATE)
                        audio.writeframes(pcm)
                check()
                if destination is not None:
                    temporary.replace(destination)
                    temporary = None
                else:
                    play_wav(temporary, timeout_s=max(.001, deadline - time.monotonic()),
                             player=self.settings.player, cancel_event=cancelled)
                    check()
            return result("completed")
        except LocalProcessCancelled:
            return result("cancelled")
        except Exception:
            return result("cancelled" if cancelled.is_set() else "failed",
                          None if cancelled.is_set() else "Gemini speech could not complete. Check connection, quota, model access, and playback.")
        finally:
            if player is not None:
                player.close()
            self._release_transports((stream, client), wait_s=.2)
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            with self._state:
                if self._active is active:
                    self._active = self._player = self._stream = self._client = None
            self._busy.release()
