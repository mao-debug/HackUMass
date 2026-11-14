"""WAV playback routing checks replace all player/process boundaries."""

import subprocess
from unittest.mock import Mock, call

import pytest

from vidi.audio import wav_playback as playback
from vidi.core.local_process import LocalProcessError


@pytest.fixture(autouse=True)
def fake_player_boundary(monkeypatch):
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, b"", b""))
    which = Mock(side_effect=lambda name: f"/mock/bin/{name}")
    monkeypatch.setattr(playback, "run_local_process", runner)
    monkeypatch.setattr(playback.shutil, "which", which)
    monkeypatch.setattr(playback.sys, "platform", "linux")
    return runner, which


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio outputs" / "voix café - final.wav"
    path.parent.mkdir()
    # Content validation belongs to the synthesizer, not this playback helper.
    path.write_bytes(b"caller-validated fixture; no real player is invoked")
    return path


def test_playback_uses_absolute_literal_unicode_path_empty_stdin_and_one_deadline(
    audio, tmp_path, monkeypatch, fake_player_boundary
):
    runner, which = fake_player_boundary
    monkeypatch.chdir(tmp_path)

    assert playback.play_wav(audio.relative_to(tmp_path), timeout_s=3.25) == "paplay"

    which.assert_called_once_with("paplay")
    runner.assert_called_once_with(
        ["/mock/bin/paplay", str(audio.resolve())], input_bytes=b"", timeout_s=3.25,
    )


def test_linux_discovers_pw_play_only_when_paplay_is_not_installed(audio, fake_player_boundary):
    runner, which = fake_player_boundary
    which.side_effect = lambda name: "/mock/bin/pw-play" if name == "pw-play" else None
    assert playback.play_wav(audio, timeout_s=2) == "pw-play"
    assert which.call_args_list == [call("paplay"), call("pw-play")]
    assert runner.call_args.args[0][0] == "/mock/bin/pw-play"


def test_mac_auto_selects_only_afplay(audio, monkeypatch, fake_player_boundary):
    runner, which = fake_player_boundary
    monkeypatch.setattr(playback.sys, "platform", "darwin")
    assert playback.play_wav(audio, timeout_s=2) == "afplay"
    which.assert_called_once_with("afplay")
    assert runner.call_args.args[0] == ["/mock/bin/afplay", str(audio.resolve())]


def test_aplay_requires_explicit_selection(audio, fake_player_boundary):
    runner, which = fake_player_boundary
    which.side_effect = lambda name: "/mock/bin/aplay" if name == "aplay" else None
    with pytest.raises(LocalProcessError, match="installed"):
        playback.play_wav(audio, timeout_s=2)
    assert which.call_args_list == [call("paplay"), call("pw-play")]
    runner.assert_not_called()
    which.reset_mock()
    assert playback.play_wav(audio, timeout_s=2, player="aplay") == "aplay"
    which.assert_called_once_with("aplay")
    runner.assert_called_once_with(["/mock/bin/aplay", str(audio.resolve())], input_bytes=b"", timeout_s=2)


@pytest.mark.parametrize("player", ["bash", "paplay --volume=5", "/tmp/player-script", "paplay;command"])
def test_arbitrary_players_paths_or_arguments_are_rejected(audio, fake_player_boundary, player):
    runner, which = fake_player_boundary
    with pytest.raises(LocalProcessError, match="Choose an installed"):
        playback.play_wav(audio, timeout_s=2, player=player)
    which.assert_not_called()
    runner.assert_not_called()


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_missing_or_nonregular_input_never_starts_a_player(tmp_path, fake_player_boundary, kind):
    runner, which = fake_player_boundary
    path = tmp_path / "missing.wav" if kind == "missing" else tmp_path
    with pytest.raises(LocalProcessError, match="existing regular file"):
        playback.play_wav(path, timeout_s=2)
    which.assert_not_called()
    runner.assert_not_called()


def test_missing_explicit_player_does_not_try_another_player(audio, fake_player_boundary):
    runner, which = fake_player_boundary
    which.return_value, which.side_effect = None, None
    with pytest.raises(LocalProcessError, match="installed"):
        playback.play_wav(audio, timeout_s=2, player="paplay")
    which.assert_called_once_with("paplay")
    runner.assert_not_called()


def test_nonzero_player_exit_is_sanitized_and_does_not_retry_or_fall_back(audio, fake_player_boundary, capsys, caplog):
    runner, which = fake_player_boundary
    runner.return_value = subprocess.CompletedProcess([], 1, b"PRIVATE output", b"PRIVATE device diagnostic")
    with pytest.raises(LocalProcessError, match="WAV playback failed") as caught:
        playback.play_wav(audio, timeout_s=2)
    assert "PRIVATE" not in str(caught.value)
    runner.assert_called_once()
    which.assert_called_once_with("paplay")
    assert capsys.readouterr() == ("", "") and caplog.text == ""


def test_shared_timeout_error_propagates_without_retry(audio, fake_player_boundary):
    runner, which = fake_player_boundary
    runner.side_effect = LocalProcessError("The local process timed out.")
    with pytest.raises(LocalProcessError, match="timed out"):
        playback.play_wav(audio, timeout_s=2)
    runner.assert_called_once()
    which.assert_called_once_with("paplay")


def test_keyboard_interrupt_propagates_after_shared_helper_cleanup(audio, fake_player_boundary):
    runner, which = fake_player_boundary
    runner.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        playback.play_wav(audio, timeout_s=2)
    runner.assert_called_once()
    which.assert_called_once_with("paplay")


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeouts_do_not_probe_or_launch_players(audio, fake_player_boundary, timeout):
    runner, which = fake_player_boundary
    with pytest.raises(LocalProcessError, match="finite positive"):
        playback.play_wav(audio, timeout_s=timeout)
    which.assert_not_called()
    runner.assert_not_called()


def test_unknown_platform_requires_explicit_player(audio, monkeypatch, fake_player_boundary):
    runner, which = fake_player_boundary
    monkeypatch.setattr(playback.sys, "platform", "unsupported")
    with pytest.raises(LocalProcessError, match="select an installed player"):
        playback.play_wav(audio, timeout_s=2)
    which.assert_not_called()
    runner.assert_not_called()
