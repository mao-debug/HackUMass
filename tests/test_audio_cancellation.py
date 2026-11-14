"""Cancellation contracts, without installed speech engines or audible output."""

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from unittest.mock import Mock, call
import wave

import pytest

from vidi.audio import piper_speech, text_to_speech, wav_playback
from vidi.core import local_process as processes
from vidi.core.configuration import SpeechSettings
from vidi.core.local_process import LocalProcessCancelled


@pytest.fixture(autouse=True)
def no_audio_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cancellation tests must not contact services or run audio engines")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(text_to_speech, "run_local_process", forbidden)
    monkeypatch.setattr(piper_speech, "run_local_process", forbidden)
    monkeypatch.setattr(piper_speech, "play_wav", forbidden)
    monkeypatch.setattr(wav_playback, "run_local_process", forbidden)


def write_wav(path):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 64)


@pytest.fixture
def piper_settings(tmp_path):
    model = tmp_path / "fake-local-voice.onnx"
    model.write_bytes(b"never loaded")
    Path(str(model) + ".json").write_text("{}")
    return SpeechSettings(piper_model_path=model, player="aplay")


def test_precancelled_process_never_spawns(monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    popen = Mock()
    monkeypatch.setattr(processes.subprocess, "Popen", popen)

    with pytest.raises(LocalProcessCancelled, match="cancelled"):
        processes.run_local_process(["never-run"], input_bytes=b"private text", timeout_s=1,
                                    cancel_event=cancelled)

    popen.assert_not_called()


def test_cancel_between_poll_intervals_terminates_group_and_does_not_resend_input(monkeypatch):
    cancelled = threading.Event()
    child = Mock(pid=12345, returncode=-signal.SIGTERM)
    child.wait.return_value = -signal.SIGTERM

    def communicate(**kwargs):
        cancelled.set()
        raise subprocess.TimeoutExpired("fake-engine", 0.05)

    child.communicate.side_effect = communicate
    monkeypatch.setattr(processes.subprocess, "Popen", Mock(return_value=child))
    kill = Mock()
    monkeypatch.setattr(processes.os, "killpg", kill)

    with pytest.raises(LocalProcessCancelled):
        processes.run_local_process(["fake-engine"], input_bytes=b"literal text", timeout_s=1,
                                    cancel_event=cancelled)

    child.communicate.assert_called_once_with(input=b"literal text", timeout=0.05)
    if os.name == "posix":
        assert kill.call_args_list == [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)]
    for pipe in (child.stdin, child.stdout, child.stderr):
        pipe.close.assert_called_once()


@pytest.mark.skipif(os.name != "posix", reason="Mac/Linux process group lifecycle")
def test_real_event_cancellation_kills_local_child_and_its_process_group(monkeypatch, tmp_path):
    """Harmless Python sleepers stand in for a synthesis/playback process tree."""
    original_popen = subprocess.Popen
    original_killpg = os.killpg
    children = []
    pidfile = tmp_path / "grandchild.pid"
    cancelled = threading.Event()
    descendant = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    script = (
        "import pathlib,signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid)); time.sleep(60)"
    )

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    signals = Mock(wraps=original_killpg)
    monkeypatch.setattr(processes.subprocess, "Popen", spawn)
    monkeypatch.setattr(processes.os, "killpg", signals)

    def cancel_when_started():
        deadline = time.monotonic() + 2
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled.set()

    timer = threading.Thread(target=cancel_when_started, daemon=True)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(LocalProcessCancelled):
            processes.run_local_process([sys.executable, "-c", script], input_bytes=b"", timeout_s=4,
                                        cancel_event=cancelled)
        assert len(children) == 1 and children[0].poll() is not None
        assert children[0].returncode < 0
        assert time.monotonic() - started < 3
        assert pidfile.exists(), "The test must actually exercise a descendant"
        signals.assert_any_call(children[0].pid, signal.SIGTERM)
        signals.assert_any_call(children[0].pid, signal.SIGKILL)
        descendant_pid = int(pidfile.read_text())
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                break
            # A reparented zombie has stopped executing and awaits the OS reaper.
            stat = Path(f"/proc/{descendant_pid}/stat")
            if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Cancellation left the descendant executing")
    finally:
        cancelled.set()
        timer.join(timeout=0.2)
        for child in children:
            try:
                original_killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=1)


@pytest.mark.parametrize("backend", ["espeak_ng", "say"])
def test_local_speech_passes_cancellation_and_returns_cancelled(monkeypatch, backend):
    event = threading.Event()
    engine = Mock(side_effect=LocalProcessCancelled("The local process was cancelled."))
    monkeypatch.setattr(text_to_speech, "run_local_process", engine)

    result = text_to_speech.LocalSpeech(SpeechSettings(), backend).speak("Read once.", cancel_event=event)

    assert result.status == "cancelled" and result.error is None
    assert engine.call_args.kwargs["cancel_event"] is event
    assert engine.call_args.kwargs["input_bytes"] == b"Read once."


@pytest.mark.parametrize("backend", ["espeak_ng", "say", "piper"])
def test_precancelled_speech_starts_no_engine(backend, piper_settings):
    event = threading.Event()
    event.set()

    result = text_to_speech.LocalSpeech(piper_settings, backend).speak("Read once.", cancel_event=event)

    assert result.status == "cancelled"


def test_local_wav_cancel_during_validation_preserves_existing_export(monkeypatch, tmp_path):
    event = threading.Event()
    destination = tmp_path / "existing.wav"
    destination.write_bytes(b"original destination")
    temporary = []

    def synthesize(argv, **kwargs):
        path = Path(argv[argv.index("-w") + 1])
        temporary.append(path)
        write_wav(path)
        assert kwargs["cancel_event"] is event
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    validate = text_to_speech._validate_wav

    def validate_then_cancel(path):
        validate(path)
        event.set()

    monkeypatch.setattr(text_to_speech, "run_local_process", synthesize)
    monkeypatch.setattr(text_to_speech, "_validate_wav", validate_then_cancel)

    result = text_to_speech.LocalSpeech(SpeechSettings(), "espeak_ng").speak(
        "Read once.", output_path=destination, cancel_event=event,
    )

    assert result.status == "cancelled"
    assert destination.read_bytes() == b"original destination"
    assert temporary and not temporary[0].exists()


def test_piper_passes_same_cancellation_to_synthesis_and_playback_and_cleans_wav(monkeypatch, piper_settings):
    event = threading.Event()
    temporary = []

    def synthesize(argv, **kwargs):
        assert kwargs["cancel_event"] is event
        path = Path(argv[argv.index("--output-file") + 1])
        temporary.append(path)
        write_wav(path)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def playback(path, **kwargs):
        assert kwargs["cancel_event"] is event
        assert path == temporary[0] and path.is_file()
        event.set()
        raise LocalProcessCancelled("The local process was cancelled.")

    synth = Mock(side_effect=synthesize)
    player = Mock(side_effect=playback)
    monkeypatch.setattr(piper_speech, "run_local_process", synth)
    monkeypatch.setattr(piper_speech, "play_wav", player)

    result = text_to_speech.LocalSpeech(piper_settings, "piper").speak("Read once.", cancel_event=event)

    assert result.status == "cancelled" and result.backend == "piper"
    synth.assert_called_once()
    player.assert_called_once()
    assert not temporary[0].exists()


def test_wav_player_forwards_cancellation_without_retry(monkeypatch, tmp_path):
    path = tmp_path / "literal audio.wav"
    path.write_bytes(b"file contents are caller validated")
    event = threading.Event()
    monkeypatch.setattr(wav_playback.shutil, "which", Mock(return_value="/already-installed/paplay"))
    player = Mock(side_effect=LocalProcessCancelled("The local process was cancelled."))
    monkeypatch.setattr(wav_playback, "run_local_process", player)

    with pytest.raises(LocalProcessCancelled):
        wav_playback.play_wav(path, timeout_s=1, player="paplay", cancel_event=event)

    player.assert_called_once_with(
        ["/already-installed/paplay", str(path.resolve())], input_bytes=b"", timeout_s=1, cancel_event=event,
    )
