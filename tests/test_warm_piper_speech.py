"""Persistent Piper behavior uses fake assets and workers, never a real model."""

from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from vidi.audio import piper_speech as speech
from vidi.core.configuration import SpeechSettings
from vidi.core.local_process import LocalProcessCancelled, LocalProcessError


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        audio.writeframes(b"\x10\x00" * 128)
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake voice never loaded")
    Path(str(model) + ".json").write_text("{}")
    python = tmp_path / "speech-python"
    python.write_bytes(b"fake runtime never executed")
    python.chmod(0o700)
    return SpeechSettings(piper_model_path=model, piper_python_path=python, timeout_s=5)


def worker():
    return type("Worker", (), {
        "next": Mock(return_value={"ready": True}),
        "synthesize": Mock(return_value=wav_bytes()), "close": Mock(),
    })()


@pytest.fixture(autouse=True)
def no_real_engines(monkeypatch):
    monkeypatch.setattr(speech.subprocess, "Popen", Mock(side_effect=AssertionError("Real synthesis is forbidden")))
    monkeypatch.setattr(speech, "play_wav", Mock(side_effect=AssertionError("Real playback is forbidden")))


def test_warm_voice_is_lazy_reused_then_closed(settings, tmp_path):
    voice = worker()
    factory = Mock(return_value=voice)
    engine = speech.WarmPiperSpeech(settings, worker_factory=factory)
    factory.assert_not_called()
    first = engine.speak("First utterance.", output_path=tmp_path / "first.wav")
    second = engine.speak("Second utterance.", output_path=tmp_path / "second.wav")
    assert first.status == second.status == "completed"
    factory.assert_called_once_with(settings.piper_python_path.absolute(), settings.piper_model_path.resolve())
    voice.next.assert_called_once()
    assert voice.synthesize.call_count == 2
    voice.close.assert_not_called()
    engine.close()
    engine.close()
    voice.close.assert_called_once()
    assert engine.speak("After close.").status == "failed"


def test_cancellation_before_capture_and_late_result_never_exports(settings, tmp_path):
    voice = worker()
    factory = Mock(return_value=voice)
    engine = speech.WarmPiperSpeech(settings, worker_factory=factory)
    event = threading.Event()
    event.set()
    assert engine.speak("Cancelled.", cancel=event).status == "cancelled"
    factory.assert_not_called()
    event.clear()
    def late(*args):
        event.set()
        return wav_bytes()
    voice.synthesize.side_effect = late
    target = tmp_path / "output.wav"
    target.write_bytes(b"previous output")
    result = engine.speak("Cancelled inference.", cancel_event=event, output_path=target)
    assert result.status == "cancelled" and target.read_bytes() == b"previous output"
    voice.close.assert_called_once()
    assert engine._worker is None


def test_stop_during_synthesis_discards_late_data(settings, tmp_path):
    voice = worker()
    engine = speech.WarmPiperSpeech(settings, worker_factory=lambda *_: voice)
    def late(*args):
        engine.stop()
        return wav_bytes()
    voice.synthesize.side_effect = late
    result = engine.speak("Test.", output_path=tmp_path / "output.wav")
    assert result.status == "cancelled" and not (tmp_path / "output.wav").exists()
    voice.close.assert_called_once()


def test_playback_cancellation_keeps_an_already_completed_warm_model(settings, monkeypatch):
    voice = worker()
    engine = speech.WarmPiperSpeech(settings, worker_factory=lambda *_: voice)
    paths = []
    def cancelled(path, **kwargs):
        paths.append(path)
        engine.stop()
        raise LocalProcessCancelled("cancelled")
    monkeypatch.setattr(speech, "play_wav", cancelled)
    result = engine.speak("Test.")
    assert result.status == "cancelled" and not paths[0].exists()
    assert engine._worker is voice
    voice.close.assert_not_called()
    engine.close()


@pytest.mark.parametrize("failure", ["startup", "inference", "invalid_wav", "oversized"])
def test_worker_failures_are_sanitized_and_unusable_worker_is_discarded(settings, tmp_path, failure, monkeypatch):
    voice = worker()
    if failure == "startup":
        voice.next.return_value = {"error": "PRIVATE"}
    elif failure == "inference":
        voice.synthesize.side_effect = LocalProcessError("PRIVATE")
    elif failure == "invalid_wav":
        voice.synthesize.return_value = b"invalid"
    else:
        monkeypatch.setattr(speech, "MAX_WARM_WAV_BYTES", 4)
        voice.synthesize.return_value = b"too large"
    engine = speech.WarmPiperSpeech(settings, worker_factory=lambda *_: voice)
    result = engine.speak("Test.", output_path=tmp_path / "voice.wav")
    assert result.status == "failed" and "PRIVATE" not in result.error
    voice.close.assert_called_once()
    assert engine._worker is None and not (tmp_path / "voice.wav").exists()


def test_missing_assets_and_protected_export_fail_before_worker(settings):
    factory = Mock()
    for selected, output in ((replace(settings, piper_model_path=None), None),
                             (settings, settings.piper_model_path),
                             (settings, settings.piper_python_path)):
        assert speech.WarmPiperSpeech(selected, worker_factory=factory).speak("Test.", output_path=output).status == "failed"
    factory.assert_not_called()


def test_warm_worker_preserves_virtual_environment_symlink(settings, tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(settings.piper_python_path)
    settings = replace(settings, piper_python_path=python)
    voice = worker()
    factory = Mock(return_value=voice)
    engine = speech.WarmPiperSpeech(settings, worker_factory=factory)
    assert engine.speak("Test.", output_path=tmp_path / "voice.wav").status == "completed"
    assert factory.call_args.args[0] == python.absolute()
    engine.close()


def test_busy_speech_is_not_queued(settings, tmp_path):
    voice = worker()
    engine = speech.WarmPiperSpeech(settings, worker_factory=lambda *_: voice)
    nested = []
    def synthesize(*args):
        nested.append(engine.speak("Second request."))
        return wav_bytes()
    voice.synthesize.side_effect = synthesize
    assert engine.speak("First request.", output_path=tmp_path / "voice.wav").status == "completed"
    assert nested[0].status == "failed" and "queued" in nested[0].error
    assert voice.synthesize.call_count == 1
    engine.close()


def test_child_wav_writer_rejects_growth_before_exceeding_bound(monkeypatch):
    observed_sizes = []

    class FakeVoice:
        @staticmethod
        def load(path):
            return FakeVoice()

        def synthesize_wav(self, text, audio, *, syn_config):
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(22050)
            try:
                audio.writeframes(b"\x01\x00" * (speech.MAX_WARM_WAV_BYTES // 2 + 1))
            finally:
                observed_sizes.append(len(audio._file.getvalue()))

    command = json.dumps({"id": "bounded", "text": "Test.", "length_scale": 1.1}) + "\n"
    output = io.StringIO()
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=FakeVoice, SynthesisConfig=lambda **_: None))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(command.encode())))
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", ["warm-worker", "fake.onnx"])
    exec(compile(speech._WARM_WORKER_SOURCE, "<isolated-worker-test>", "exec"), {})
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert replies == [{"ready": True}, {"error": "unavailable"}]
    assert observed_sizes and observed_sizes[0] <= speech.MAX_WARM_WAV_BYTES


def test_worker_constructor_rolls_back_child_when_reader_cannot_start(monkeypatch, settings):
    process = SimpleNamespace(pid=721337, stdin=io.BytesIO(), stdout=io.BytesIO(), wait=Mock(return_value=0))
    popen = Mock(return_value=process)
    kill = Mock()
    monkeypatch.setattr(speech.subprocess, "Popen", popen)
    monkeypatch.setattr(speech.os, "killpg", kill)
    monkeypatch.setattr(speech.threading.Thread, "start", Mock(side_effect=RuntimeError("thread unavailable")))
    with pytest.raises(RuntimeError, match="thread unavailable"):
        speech._WarmWorker(settings.piper_python_path, settings.piper_model_path)
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.args[0][0] == str(settings.piper_python_path)
    assert kill.call_count == 2 and process.wait.call_count == 2
    assert process.stdin.closed and process.stdout.closed
