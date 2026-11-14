"""Optional Piper behavior using local fake assets, processes, and playback."""

import builtins
from dataclasses import replace
from pathlib import Path
import stat
from subprocess import CompletedProcess
import sys
from unittest.mock import Mock
import wave

import pytest

from vidi.audio import piper_speech
from vidi.core.configuration import SpeechSettings
from vidi.core.local_process import LocalProcessError


def write_wav(path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        audio.writeframes(b"\0\0" * 128)


@pytest.fixture
def settings(tmp_path):
    model = tmp_path / "private-voices" / "demo-voice.onnx"
    model.parent.mkdir()
    model.write_bytes(b"fake local model, never loaded")
    Path(str(model) + ".json").write_text("{}", encoding="utf-8")
    python = tmp_path / "private-runtime" / "python"
    python.parent.mkdir()
    python.write_bytes(b"fake executable, never run")
    python.chmod(0o755)
    return SpeechSettings(
        piper_model_path=model, piper_python_path=python, piper_length_scale=1.25,
        player="aplay", timeout_s=10,
    )


@pytest.fixture
def boundaries(monkeypatch):
    paths = []

    def synthesize(argv, *, input_bytes, timeout_s):
        path = Path(argv[argv.index("--output-file") + 1])
        paths.append(path)
        write_wav(path)
        return CompletedProcess(argv, 0, b"", b"private stderr must stay private")

    synth = Mock(side_effect=synthesize)
    player = Mock(return_value="aplay")
    monkeypatch.setattr(piper_speech, "run_local_process", synth)
    monkeypatch.setattr(piper_speech, "play_wav", player)
    return synth, player, paths


def test_constructor_does_not_import_piper_or_start_any_process(monkeypatch, settings, boundaries):
    original_import = builtins.__import__

    def no_piper_import(name, *args, **kwargs):
        if name == "piper" or name.startswith("piper."):
            pytest.fail("Piper Python APIs must not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_piper_import)
    adapter = piper_speech.PiperSpeech(settings)
    assert adapter.settings is settings
    boundaries[0].assert_not_called()
    boundaries[1].assert_not_called()


def test_export_uses_fixed_cli_literal_stdin_and_atomic_sibling_wav(settings, boundaries, tmp_path):
    synth, player, paths = boundaries
    destination = tmp_path / "export" / "reading.wav"
    raw = "Hello $(touch pwned) [[ipa]] --output-file other.wav"
    result = piper_speech.PiperSpeech(settings).speak(raw, output_path=destination)
    assert result.status == "completed" and result.mode == "wav"
    assert result.voice == "demo-voice" and result.model == "demo-voice.onnx"
    assert result.output_path == destination
    argv = synth.call_args.args[0]
    assert argv == [
        str(settings.piper_python_path.resolve()), "-m", "piper", "--model", str(settings.piper_model_path.resolve()),
        "--output-file", str(paths[0]), "--length-scale", "1.25",
    ]
    assert synth.call_args.kwargs["input_bytes"] == b"Hello $(touch pwned) [ [ipa] ] --output-file other.wav"
    assert raw not in argv and "150" not in argv  # Piper receives length scale, never a WPM claim.
    assert paths[0].parent == destination.parent and not paths[0].exists()
    piper_speech._validate_wav(destination)
    player.assert_not_called()


def test_playback_uses_one_private_validated_wav_then_removes_it(settings, boundaries):
    synth, player, paths = boundaries

    def play(path, *, timeout_s, player):
        assert path == paths[0] and path.is_file()
        piper_speech._validate_wav(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert 0 < timeout_s <= settings.timeout_s and player == "aplay"
        return "aplay"

    player.side_effect = play
    result = piper_speech.PiperSpeech(settings).speak("Read this once.")
    assert result.status == "completed" and result.mode == "playback" and result.output_path is None
    synth.assert_called_once()
    player.assert_called_once()
    assert not paths[0].exists()


@pytest.mark.parametrize("missing", ["unconfigured", "model", "companion", "python", "nonexecutable"])
def test_missing_local_assets_fail_without_synthesis_or_playback(settings, boundaries, tmp_path, missing):
    if missing == "unconfigured":
        settings = replace(settings, piper_model_path=None)
    elif missing == "model":
        settings = replace(settings, piper_model_path=tmp_path / "absent.onnx")
    elif missing == "companion":
        Path(str(settings.piper_model_path) + ".json").unlink()
    elif missing == "python":
        settings = replace(settings, piper_python_path=tmp_path / "absent-python")
    else:
        settings.piper_python_path.chmod(0o644)
    result = piper_speech.PiperSpeech(settings).speak("Read this.")
    assert result.status == "failed" and str(tmp_path) not in result.error
    boundaries[0].assert_not_called()
    boundaries[1].assert_not_called()


@pytest.mark.parametrize("target", ["model", "companion", "model_alias", "python_alias"])
def test_export_cannot_replace_voice_assets_or_runtime(settings, boundaries, tmp_path, target):
    protected = settings.piper_model_path if target != "companion" else Path(str(settings.piper_model_path) + ".json")
    if target == "python_alias":
        protected = settings.piper_python_path
    before = protected.read_bytes()
    destination = protected
    if target.endswith("alias"):
        destination = tmp_path / "alias.wav"
        destination.symlink_to(protected)
    result = piper_speech.PiperSpeech(settings).speak("Read this.", output_path=destination)
    assert result.status == "failed" and protected.read_bytes() == before
    boundaries[0].assert_not_called()
    boundaries[1].assert_not_called()


def test_default_python_and_shared_text_preparation_are_used(settings, boundaries, tmp_path):
    synth, _, _ = boundaries
    settings = replace(settings, piper_python_path=None, max_words=2)
    result = piper_speech.PiperSpeech(settings).speak("one two three", output_path=tmp_path / "short.wav")
    assert result.status == "completed" and result.truncated is True
    assert synth.call_args.args[0][0] == sys.executable
    assert synth.call_args.kwargs["input_bytes"] == b"one two More text is available."


def test_configured_venv_python_symlink_is_preserved_in_command(settings, boundaries, tmp_path):
    python = tmp_path / "piper-environment" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(settings.piper_python_path)
    settings = replace(settings, piper_python_path=python)
    result = piper_speech.PiperSpeech(settings).speak("Read this.", output_path=tmp_path / "reading.wav")
    assert result.status == "completed"
    command = boundaries[0].call_args.args[0]
    assert command[0] == str(python.absolute())
    assert command[0] != str(python.resolve())


def test_one_deadline_is_shared_across_synthesis_validation_and_playback(monkeypatch, settings, boundaries):
    synth, player, _ = boundaries
    clock = [0.0]
    monkeypatch.setattr(piper_speech.time, "monotonic", lambda: clock[0])
    synthesize = synth.side_effect
    validate = piper_speech._validate_wav

    def generate(*args, **kwargs):
        assert kwargs["timeout_s"] == 10.0
        clock[0] += 3.0
        return synthesize(*args, **kwargs)

    def inspect(path):
        validate(path)
        clock[0] += 2.0

    def play(path, *, timeout_s, player):
        assert timeout_s == 5.0
        clock[0] += 1.0
        return "aplay"

    synth.side_effect = generate
    player.side_effect = play
    monkeypatch.setattr(piper_speech, "_validate_wav", inspect)
    result = piper_speech.PiperSpeech(settings).speak("Read this.")
    assert result.status == "completed" and result.elapsed_s == 6.0


@pytest.mark.parametrize("stage", ["synthesis", "validation"])
def test_expired_budget_never_starts_playback(monkeypatch, settings, boundaries, stage):
    synth, player, paths = boundaries
    clock = [0.0]
    monkeypatch.setattr(piper_speech.time, "monotonic", lambda: clock[0])
    generate = synth.side_effect
    validate = piper_speech._validate_wav

    def synthesize(*args, **kwargs):
        completed = generate(*args, **kwargs)
        if stage == "synthesis":
            clock[0] = 10.0
        return completed

    def inspect(path):
        validate(path)
        clock[0] = 10.0

    synth.side_effect = synthesize
    monkeypatch.setattr(piper_speech, "_validate_wav", inspect)
    result = piper_speech.PiperSpeech(settings).speak("Read this.")
    assert result.status == "failed"
    player.assert_not_called()
    assert not paths[0].exists()


def test_playback_failure_removes_private_audio_without_retry_or_diagnostic_leak(settings, boundaries):
    synth, player, paths = boundaries
    player.side_effect = LocalProcessError("private output-device diagnostics")
    result = piper_speech.PiperSpeech(settings).speak("Read this.")
    assert result.status == "failed" and "private output-device" not in result.error
    assert not paths[0].exists()
    synth.assert_called_once()
    player.assert_called_once()


@pytest.mark.parametrize("failure", ["exit", "invalid_wav", "process"])
def test_failed_export_preserves_old_file_and_hides_private_diagnostics(settings, boundaries, tmp_path, failure, capsys):
    synth, player, _ = boundaries
    destination = tmp_path / "reading.wav"
    destination.write_bytes(b"existing audio must survive")
    secret = f"private diagnostics {settings.piper_model_path}"
    if failure == "exit":
        synth.side_effect = lambda argv, **kwargs: CompletedProcess(argv, 2, b"", secret.encode())
    elif failure == "invalid_wav":
        synth.side_effect = lambda argv, **kwargs: CompletedProcess(argv, 0, b"", secret.encode())
    else:
        synth.side_effect = LocalProcessError(secret)
    result = piper_speech.PiperSpeech(settings).speak("Read this.", output_path=destination)
    assert result.status == "failed" and secret not in result.error and str(tmp_path) not in result.error
    assert destination.read_bytes() == b"existing audio must survive"
    assert list(tmp_path.glob(".vidi-piper-*.wav")) == []
    synth.assert_called_once()
    player.assert_not_called()
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


@pytest.mark.parametrize("stage", ["synthesis", "validation", "playback"])
def test_cancellation_propagates_after_temp_cleanup_and_preserves_existing_export(monkeypatch, settings, boundaries, tmp_path, stage):
    synth, player, paths = boundaries
    destination = tmp_path / "reading.wav"
    destination.write_bytes(b"old output")
    if stage == "synthesis":
        def interrupted(argv, **kwargs):
            paths.append(Path(argv[argv.index("--output-file") + 1]))
            raise KeyboardInterrupt
        synth.side_effect = interrupted
    elif stage == "validation":
        monkeypatch.setattr(piper_speech, "_validate_wav", Mock(side_effect=KeyboardInterrupt))
    else:
        player.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        piper_speech.PiperSpeech(settings).speak("Read this.", output_path=None if stage == "playback" else destination)
    assert destination.read_bytes() == b"old output"
    assert paths and not paths[0].exists()
