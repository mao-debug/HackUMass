"""Speech tests never invoke an audio engine, a speaker, or a network service."""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess
from unittest.mock import Mock
import wave

import pytest

from vidi.audio import text_to_speech as speech
from vidi.core.configuration import SpeechSettings
from vidi.core.local_process import LocalProcessError


@pytest.fixture(autouse=True)
def block_real_engine(monkeypatch):
    """Every engine boundary is mocked, including tests for early failures."""
    helper = Mock(return_value=subprocess.CompletedProcess([], 0, b"", b""))
    monkeypatch.setattr(speech, "run_local_process", helper)
    return helper


def write_pcm(path: Path, frames: int = 3) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        audio.writeframes(struct.pack("<h", 300) * frames)


def test_espeak_uses_literal_utf8_stdin_and_fixed_arguments(block_real_engine):
    text = 'Read "café"; $(touch file) --voice=other <speak>hello</speak>'
    result = speech.LocalSpeech(SpeechSettings(), "espeak_ng").speak(text)

    assert result.status == "completed"
    assert result.mode == "playback"
    assert result.backend == "espeak_ng"
    assert result.text == text
    assert result.output_path is None
    assert result.truncated is False
    block_real_engine.assert_called_once_with(
        ["espeak-ng", "-v", "en", "-s", "150", "--stdin"],
        input_bytes=text.encode("utf-8"), timeout_s=45.0,
    )


@pytest.mark.parametrize("backend", ["say", "espeak_ng"])
def test_speech_directives_and_controls_are_neutralized(backend, block_real_engine):
    text = "Hello\x00\n[[rate 1]][[[volm 0]]]\x1bworld"
    result = speech.LocalSpeech(SpeechSettings(), backend).speak(text)

    assert result.status == "completed"
    assert "[[" not in result.text
    assert "]]" not in result.text
    assert "\x00" not in result.text and "\x1b" not in result.text
    assert "rate 1" in result.text and "volm 0" in result.text
    assert text == "Hello\x00\n[[rate 1]][[[volm 0]]]\x1bworld"
    assert block_real_engine.call_args.kwargs["input_bytes"] == result.text.encode("utf-8")


def test_say_uses_selected_voice_and_rate(block_real_engine):
    result = speech.LocalSpeech(SpeechSettings(speed_wpm=180, voice="Alex"), "say").speak("Hello.")
    assert result.status == "completed"
    block_real_engine.assert_called_once_with(
        ["say", "-v", "Alex", "-r", "180", "-f", "-"],
        input_bytes=b"Hello.", timeout_s=45.0,
    )


@pytest.mark.parametrize("platform, expected", [("darwin", "say"), ("linux", "espeak_ng")])
def test_platform_default_playback(platform, expected, monkeypatch, block_real_engine):
    monkeypatch.setattr(speech.sys, "platform", platform)
    result = speech.LocalSpeech(SpeechSettings()).speak("Hello.")
    assert result.backend == expected
    assert result.status == "completed"
    assert block_real_engine.call_count == 1
    if expected == "say":
        assert block_real_engine.call_args.args[0] == ["say", "-v", "Samantha", "-r", "150", "-f", "-"]


def test_unsupported_platform_returns_failure(monkeypatch, block_real_engine):
    monkeypatch.setattr(speech.sys, "platform", "win32")
    result = speech.LocalSpeech(SpeechSettings()).speak("Hello.")
    assert result.status == "failed"
    assert result.backend is None
    block_real_engine.assert_not_called()


@pytest.mark.parametrize("text", ["", " \t\n", "\x00\x1b"])
def test_empty_text_speaks_fixed_feedback_once(text, block_real_engine):
    result = speech.LocalSpeech(SpeechSettings(), "espeak_ng").speak(text)
    assert result.text == "No readable text found."
    assert result.truncated is False
    block_real_engine.assert_called_once()


@pytest.mark.parametrize("word_count, truncated", [(60, False), (61, True)])
def test_recognized_words_are_bounded_without_losing_source(word_count, truncated, block_real_engine):
    text = " ".join(f"word{index}" for index in range(word_count))
    result = speech.LocalSpeech(SpeechSettings(), "espeak_ng").speak(text)
    assert result.truncated is truncated
    expected = " ".join(f"word{index}" for index in range(60))
    if truncated:
        expected += " More text is available."
    assert result.text == expected
    assert len(text.split()) == word_count
    assert block_real_engine.call_count == 1


def test_word_limit_preserves_printable_spacing_and_punctuation(block_real_engine):
    text = 'Read  "this",   please. And keep the rest in OCR.'
    result = speech.LocalSpeech(SpeechSettings(max_words=3), "espeak_ng").speak(text)
    assert result.text == 'Read  "this",   please. More text is available.'
    assert result.truncated is True


@pytest.mark.parametrize("failure", [LocalProcessError("engine missing"), LocalProcessError("timeout")])
def test_helper_failures_are_observable_and_not_retried(failure, block_real_engine):
    block_real_engine.side_effect = failure
    result = speech.LocalSpeech(SpeechSettings(), "espeak_ng").speak("Keep this result.")
    assert result.status == "failed"
    assert result.error
    assert result.text == "Keep this result."
    block_real_engine.assert_called_once()


def test_nonzero_engine_failure_does_not_echo_untrusted_stderr(block_real_engine):
    block_real_engine.return_value = subprocess.CompletedProcess([], 2, b"", b"private recognized text")
    result = speech.LocalSpeech(SpeechSettings(), "espeak_ng").speak("Text.")
    assert result.status == "failed"
    assert "status 2" in result.error
    assert "private recognized text" not in result.error


def test_wav_export_is_espeak_only_and_atomically_replaces_file(tmp_path, monkeypatch, block_real_engine):
    monkeypatch.setattr(speech.sys, "platform", "darwin")
    destination = tmp_path / "new" / "speech.wav"
    destination.parent.mkdir()
    destination.write_bytes(b"previous recording")
    observed_paths = []

    def synthesize(argv, *, input_bytes, timeout_s):
        assert argv[0] == "espeak-ng"
        assert "--stdin" in argv
        assert "-w" in argv
        path = Path(argv[argv.index("-w") + 1])
        assert path != destination and path.parent == destination.parent
        assert destination.read_bytes() == b"previous recording"
        observed_paths.append(path)
        write_pcm(path)
        assert destination.read_bytes() == b"previous recording"
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    block_real_engine.side_effect = synthesize
    result = speech.LocalSpeech(SpeechSettings()).speak("Export only.", output_path=destination)
    assert result.status == "completed"
    assert result.mode == "wav" and result.backend == "espeak_ng"
    assert result.output_path == destination
    with wave.open(str(destination), "rb") as audio:
        assert audio.getnframes() == 3
    assert all(not path.exists() for path in observed_paths)
    block_real_engine.assert_called_once()


@pytest.mark.parametrize("failure", ["nonzero", "empty", "not-wave", "non-pcm", "zero-frames", "incomplete", "helper", "interrupt"])
def test_failed_wav_export_preserves_destination_and_cleans_temp(failure, tmp_path, block_real_engine):
    destination = tmp_path / "speech.wav"
    destination.write_bytes(b"previous recording")
    observed = []

    def synthesize(argv, *, input_bytes, timeout_s):
        temporary = Path(argv[argv.index("-w") + 1])
        observed.append(temporary)
        if failure == "helper":
            raise LocalProcessError("engine missing")
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "not-wave":
            temporary.write_bytes(b"not a wave file")
        elif failure == "non-pcm":
            temporary.write_bytes(
                b"RIFF" + struct.pack("<I", 40) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 3, 1, 22050, 88200, 4, 32)
                + b"data" + struct.pack("<I", 4) + b"\x00" * 4
            )
        elif failure == "zero-frames":
            write_pcm(temporary, frames=0)
        elif failure == "incomplete":
            write_pcm(temporary)
            temporary.write_bytes(temporary.read_bytes()[:-1])
        return subprocess.CompletedProcess(argv, 1 if failure == "nonzero" else 0, b"", b"")

    block_real_engine.side_effect = synthesize
    speaker = speech.LocalSpeech(SpeechSettings(), "espeak_ng")
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            speaker.speak("Preserve existing output.", output_path=destination)
    else:
        result = speaker.speak("Preserve existing output.", output_path=destination)
        assert result.status == "failed"
        assert result.error
    assert destination.read_bytes() == b"previous recording"
    assert all(not path.exists() for path in observed)
    assert list(tmp_path.iterdir()) == [destination]
    block_real_engine.assert_called_once()


def test_say_wav_is_rejected_without_engine_call(tmp_path, block_real_engine):
    result = speech.LocalSpeech(SpeechSettings(), "say").speak("Text", output_path=tmp_path / "out.wav")
    assert result.status == "failed"
    assert "requires espeak_ng" in result.error
    assert not list(tmp_path.iterdir())
    block_real_engine.assert_not_called()


def test_playback_interrupt_is_propagated_without_retry(block_real_engine):
    block_real_engine.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        speech.LocalSpeech(SpeechSettings(), "say").speak("Stop.")
    block_real_engine.assert_called_once()


def test_wav_destination_error_is_reported_before_engine(tmp_path, block_real_engine):
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_bytes(b"keep")
    result = speech.LocalSpeech(SpeechSettings(), "espeak_ng").speak("Text", output_path=parent_file / "out.wav")
    assert result.status == "failed"
    assert parent_file.read_bytes() == b"keep"
    block_real_engine.assert_not_called()
