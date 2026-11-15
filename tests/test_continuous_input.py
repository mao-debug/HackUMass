"""Continuous input tests use in-memory pipes; they never open a microphone."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import io
import subprocess
import threading
import time
import wave

import pytest

from vidi.audio import continuous_input as audio
from vidi.audio.speech_to_text import SpeechRecognitionError
from vidi.core.local_process import LocalProcessCancelled


def eventually(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("The fake audio worker did not reach the expected state")
        threading.Event().wait(.001)


class Pipe:
    def __init__(self):
        self.condition = threading.Condition()
        self.chunks = deque()
        self.closed = False

    def push(self, data):
        with self.condition:
            self.chunks.append(data)
            self.condition.notify_all()

    def read(self, size):
        with self.condition:
            while not self.chunks and not self.closed:
                self.condition.wait(.05)
            return self.chunks.popleft() if self.chunks else b""

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class Process:
    def __init__(self):
        self.stdout = Pipe()
        self.stopped = False


@pytest.fixture
def processes(monkeypatch):
    calls, made = [], []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        made.append(Process())
        return made[-1]

    def stop(process):
        process.stopped = True
        if process.stdout is not None:
            process.stdout.close()

    monkeypatch.setattr(audio, "_stop_process", stop)
    monkeypatch.setattr(audio, "_close_pipes", lambda process: None)
    return factory, calls, made


def test_arecord_literal_arguments_restart_and_sample_alignment(processes):
    factory, calls, made = processes
    source = audio.AudioInput("literal;not-a-shell", process_factory=factory, source_validator=lambda _: None)
    source.start()
    source.start()
    args, kwargs = calls[0]
    assert len(calls) == 1
    assert args[0] == ["arecord", "--quiet", "--device", "literal;not-a-shell", "--file-type", "raw",
                       "--format", "S16_LE", "--rate", "16000", "--channels", "1"]
    assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["shell"] is False
    made[0].stdout.push(b"\x01")
    made[0].stdout.push(b"\x02\x03")
    assert source.read() == b"\x01\x02"
    made[0].stdout.push(b"\x04")
    assert source.read() == b"\x03\x04"
    assert source.read(0) == b""
    made[0].stdout.push(b"\x05\x06")
    eventually(lambda: source.buffered_bytes == 2)
    source.close()
    assert made[0].stopped and source.buffered_bytes == 0
    with pytest.raises(SpeechRecognitionError, match="not running"):
        source.read(0)
    source.start()
    assert len(calls) == 2 and source.read(0) == b""
    made[1].stdout.push(b"\x07\x08")
    assert source.read() == b"\x07\x08"
    source.close()
    source.close()


def test_pipe_buffer_is_bounded_and_disconnection_discards_pending_pcm(processes):
    factory, _, made = processes
    source = audio.AudioInput("verified", process_factory=factory, source_validator=lambda _: None)
    source.start()
    for _ in range(300):
        made[0].stdout.push(b"\0" * audio.CHUNK_BYTES)
    eventually(lambda: source.dropped_bytes > 0)
    assert source.buffered_bytes <= audio.MAX_BUFFER_BYTES
    made[0].stdout.close()
    eventually(lambda: source._error is not None)
    with pytest.raises(SpeechRecognitionError, match="headset connection"):
        source.read()
    assert source.buffered_bytes == 0
    source.close()


def test_partial_setup_process_is_cleaned_up(processes):
    factory, _, made = processes

    def broken(*args, **kwargs):
        process = factory(*args, **kwargs)
        process.stdout = None
        return process

    source = audio.AudioInput("verified", process_factory=broken, source_validator=lambda _: None)
    with pytest.raises(SpeechRecognitionError, match="could not start"):
        source.start()
    assert made[0].stopped


def test_reader_thread_start_failure_releases_recorder(processes,monkeypatch):
    factory,_,made = processes

    def fail(_):
        raise RuntimeError("private startup details")

    source = audio.AudioInput("verified",process_factory=factory,source_validator=lambda _:None)
    with monkeypatch.context() as failed:
        failed.setattr(threading.Thread,"start",fail)
        with pytest.raises(SpeechRecognitionError,match="could not start") as error:
            source.start()
    assert made[0].stopped and source._process is None and source._thread is None
    assert "private" not in str(error.value)
    try:
        source.start()
        assert len(made) == 2 and source.read(0) == b""
        made[1].stdout.push(b"\x01\x02")
        assert source.read() == b"\x01\x02"
    finally:
        source.close()


@pytest.mark.parametrize("device,rate", [("",16000), ("pulse\nother",16000), ("pulse",8000), ("pulse",True)])
def test_input_rejects_invalid_settings(device, rate):
    with pytest.raises(SpeechRecognitionError):
        audio.AudioInput(device, rate)


def test_default_pulse_monitor_is_rejected_before_start(monkeypatch):
    monkeypatch.setenv("PULSE_SOURCE", "output.monitor")
    source = audio.AudioInput(process_factory=lambda *a, **k: pytest.fail("No recorder is allowed"))
    with pytest.raises(SpeechRecognitionError, match="output monitor"):
        source.start()


class QueueInput:
    def __init__(self):
        self.condition = threading.Condition()
        self.queue = deque()
        self.running = False
        self.starts = self.closes = 0

    def start(self):
        with self.condition:
            self.running = True
            self.queue.clear()
            self.starts += 1

    def push(self, pcm):
        with self.condition:
            self.queue.append(pcm)
            self.condition.notify_all()

    def read(self, timeout_s=.1):
        with self.condition:
            if not self.queue and self.running:
                self.condition.wait(timeout_s)
            if not self.running:
                raise SpeechRecognitionError("The fake input stopped.")
            item = self.queue.popleft() if self.queue else b""
            if isinstance(item, Exception):
                raise item
            return item

    def close(self):
        with self.condition:
            self.closes += 1
            self.running = False
            self.queue.clear()
            self.condition.notify_all()


class Detector:
    def __init__(self):
        self.seen = []
        self.resets = 0
        self.closed = threading.Event()
        self.hit = b"\x01\0"

    def accept(self, pcm):
        self.seen.append(pcm)
        return pcm.startswith(self.hit)

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed.set()


def listener_pair(**kwargs):
    source, detector = QueueInput(), Detector()
    listener = audio.ContinuousListener(source, detector, **kwargs)
    return listener, source, detector


def test_wake_runs_during_playback_and_capture_is_owned_bounded_wav():
    listener, source, detector = listener_pair(pre_roll_s=.01)
    woken = threading.Event()
    # The input loop has no playback suppression: callback can stop active TTS.
    speaking = threading.Event()
    speaking.set()

    def wake():
        speaking.clear()
        woken.set()

    listener.start(wake)
    try:
        source.push(b"\x01\0" * 160)
        assert woken.wait(1) and not speaking.is_set()
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(listener.capture_next, .02)
            eventually(lambda: listener._capture is not None)
            source.push(b"\x02\0" * 160)
            clip = result.result(1)
        with wave.open(io.BytesIO(clip.wav_bytes), "rb") as wav:
            assert (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes()) == (1,2,16000,320)
            assert wav.readframes(320) == b"\x01\0" * 160 + b"\x02\0" * 160
        assert clip.duration_s == .02 and "wav_bytes" not in repr(clip)
        assert len(detector.seen) == 2
    finally:
        listener.close()
    assert detector.closed.wait(1)


def test_no_recursive_wakes_for_final_command_chunk_and_one_capture_only():
    listener, source, detector = listener_pair(pre_roll_s=0, cooldown_s=0)
    hits = []
    listener.start(lambda: hits.append(True))
    try:
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(listener.capture_next, .01)
            eventually(lambda: listener._capture is not None)
            with pytest.raises(SpeechRecognitionError, match="already active"):
                listener.capture_next(.01)
            source.push(b"\x01\0" * 160)
            result.result(1)
        eventually(lambda: len(detector.seen) == 1)
        assert hits == []
        source.push(b"\x01\0")
        eventually(lambda: len(hits) == 1)
    finally:
        listener.close()


@pytest.mark.parametrize("operation", ["pause", "close", "cancel"])
def test_capture_cancel_or_privacy_pause_clears_pcm(operation):
    listener, source, _ = listener_pair()
    listener.start(lambda: None)
    cancel = threading.Event()
    try:
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(listener.capture_next, 1, cancel)
            eventually(lambda: listener._capture is not None)
            source.push(b"\x02\0" * 160)
            eventually(lambda: len(listener._capture.data) > 0)
            if operation == "cancel":
                cancel.set()
            else:
                getattr(listener, operation)()
                assert not source.running and not listener._ring
                if listener._capture is not None:
                    assert not listener._capture.data
            with pytest.raises(LocalProcessCancelled):
                result.result(1)
        if operation == "pause":
            listener.resume()
            assert source.starts == 2 and source.running and not listener._ring
    finally:
        listener.close()


def test_cancelled_prefilled_clip_does_not_escape():
    listener, source, _ = listener_pair(pre_roll_s=.01)
    woken = threading.Event()
    listener.start(woken.set)
    try:
        source.push(b"\x01\0" * 160)
        assert woken.wait(1)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(LocalProcessCancelled):
            listener.capture_next(.01, cancel)
    finally:
        listener.close()


def test_listener_errors_pause_hardware_and_never_publish_raw_failure():
    listener, source, _ = listener_pair()
    listener.start(lambda: None)
    try:
        source.push(RuntimeError("private raw input must not escape"))
        eventually(lambda: listener.paused)
        assert listener.error == "Continuous microphone or wake detection is unavailable."
        assert not source.running and not listener._ring
    finally:
        listener.close()


def test_model_preparation_failure_never_opens_microphone():
    from vidi.audio.wake_word import WakeWordError
    listener, source, detector = listener_pair()

    def unavailable():
        raise WakeWordError("Wake model files are missing.")

    detector.prepare = unavailable
    try:
        with pytest.raises(WakeWordError, match="missing"):
            listener.start(lambda: None)
        assert source.starts == 0 and listener.paused
    finally:
        listener.close()


def test_listener_thread_start_failure_closes_input_and_resume_retries(monkeypatch):
    listener, source, detector = listener_pair()
    woken = threading.Event()

    def fail(_):
        raise RuntimeError("private thread failure details")

    with monkeypatch.context() as failed:
        failed.setattr(threading.Thread,"start",fail)
        with pytest.raises(SpeechRecognitionError,match="could not start") as error:
            listener.start(woken.set)
    assert "private" not in str(error.value)
    assert listener.paused and listener._thread is None
    assert not source.running and source.closes == 1 and not listener._ring
    try:
        listener.resume()
        assert source.starts == 2 and listener.error is None
        source.push(b"\x01\0")
        assert woken.wait(1)
    finally:
        listener.close()
    assert detector.closed.wait(1)


def test_partial_input_start_failure_is_closed_and_retryable(monkeypatch):
    listener, source, _ = listener_pair()
    original = source.start

    def fail():
        original()
        raise RuntimeError("private device details")

    with monkeypatch.context() as failed:
        failed.setattr(source,"start",fail)
        with pytest.raises(SpeechRecognitionError,match="could not start"):
            listener.start(lambda:None)
    assert not source.running and listener.paused and listener._thread is None
    try:
        listener.resume()
        assert source.running and not listener.paused
    finally:
        listener.close()


def test_cleanup_thread_start_failure_uses_nonblocking_detector_close(monkeypatch):
    listener, source, detector = listener_pair()
    detector.request_close = detector.closed.set

    def fail(_):
        raise RuntimeError("no more local threads")

    monkeypatch.setattr(threading.Thread,"start",fail)
    listener.close()
    assert listener.paused and not source.running and detector.closed.is_set()
    listener.close()


def test_input_overrun_fails_closed_instead_of_waking_on_delayed_pcm():
    listener, source, _ = listener_pair()
    hits = []
    source.dropped_bytes = 0
    listener.start(lambda: hits.append(True))
    try:
        source.dropped_bytes = 640
        source.push(b"\x01\0" * 160)
        eventually(lambda: listener.paused)
        assert not hits and "fell behind" in listener.error
        assert not source.running and not listener._ring
    finally:
        listener.close()


def test_unstarted_resume_does_not_open_input_and_callback_cannot_capture():
    listener, source, _ = listener_pair()
    with pytest.raises(SpeechRecognitionError, match="Start"):
        listener.resume()
    assert source.starts == 0
    errors = []

    def callback():
        try:
            listener.capture_next(.01)
        except SpeechRecognitionError as error:
            errors.append(str(error))

    listener.start(callback)
    try:
        source.push(b"\x01\0")
        eventually(lambda: bool(errors))
        assert "outside the wake callback" in errors[0]
    finally:
        listener.close()


@pytest.mark.parametrize("duration", [0,-1,5.1,True,float("inf")])
def test_command_duration_is_bounded(duration):
    listener, _, _ = listener_pair()
    with pytest.raises(SpeechRecognitionError, match="at most five"):
        listener.capture_next(duration)
