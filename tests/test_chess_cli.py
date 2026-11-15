"""Chess opt-ins and routing never access actual cameras, keys, or services."""

import builtins
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

import vidi.main as cli
from vidi.core import controller
from vidi.core.configuration import AppSettings, ChessSettings, SimulationSettings


CHESS = ["--chess-live", "--continuous", "--preview"]


@pytest.fixture
def route(monkeypatch):
    run = Mock(return_value=SimpleNamespace(interrupted=False))
    monkeypatch.setattr(controller, "run_simulation", run)
    monkeypatch.setattr(cli, "saved_gemini_key", Mock(return_value=nullcontext()))
    monkeypatch.setattr(cli, "prompted_gemini_key", Mock(return_value=nullcontext()))
    return run


def forbid_keys(monkeypatch):
    for name in ("saved_gemini_key", "prompted_gemini_key"):
        monkeypatch.setattr(cli, name, lambda: pytest.fail("Credential helper accessed"))


def test_gemini_chess_routes_only_the_requested_new_flag(route):
    assert cli.main(CHESS) == 0
    assert route.call_args.kwargs == {"continuous": True, "chess_live": True, "speak": False}
    cli.saved_gemini_key.assert_called_once_with()
    cli.prompted_gemini_key.assert_not_called()


def test_fixture_override_uses_cwd_orientation_and_never_keys(route, monkeypatch, tmp_path):
    forbid_keys(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert cli.main(CHESS + ["--chess-fixture", "board.json", "--board-orientation", "black_near", "--speak"]) == 0
    settings = route.call_args.args[0]
    assert settings.chess == ChessSettings(backend="fixture", fixture_path=tmp_path / "board.json", orientation="black_near")
    assert route.call_args.kwargs["speak"] is True
    assert "voice_commands" not in route.call_args.kwargs


@pytest.mark.parametrize("offline", [False, True])
def test_configured_fixture_always_bypasses_credentials(route, monkeypatch, tmp_path, offline):
    forbid_keys(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "FAKE-UNUSED-KEY")
    monkeypatch.setattr(cli, "load_settings", lambda _: AppSettings(
        chess=ChessSettings(backend="fixture", fixture_path=tmp_path / "board.json")))
    assert cli.main(CHESS + (["--offline"] if offline else [])) == 0
    assert route.call_args.kwargs.get("offline", False) is offline


@pytest.mark.parametrize("arguments", [
    ["--chess-live"], ["--chess-live", "--continuous"], ["--chess-live", "--preview"],
    ["--chess-fixture", "board.json"], ["--board-orientation", "white_near"],
    CHESS + ["--scene-live"], CHESS + ["--describe"], CHESS + ["--read-text"],
    CHESS + ["--gemini-detect"], CHESS + ["--gemini-live"], CHESS + ["--gemini-read"],
    CHESS + ["--proximity"], CHESS + ["--sensor-port", "/dev/never-open"],
    CHESS + ["--question", "What should I play?"], CHESS + ["--speech-output", "board.wav"],
    CHESS + ["--offline"], CHESS + ["--detector", "fixture"], CHESS + ["--source", "imx500"],
    CHESS + ["--max-frames", "5"],
    CHESS + ["--chess-fixture", "board.json", "--voice-commands"],
    CHESS + ["--chess-fixture", "board.json", "--prompt-key"],
    CHESS + ["--chess-fixture", "board.json", "--report", "board.json"],
    CHESS + ["--image", "board.png", "--output", "board.png"],
    CHESS + ["--voice-commands", "--command-audio", "command.wav", "--report", "command.wav"],
])
def test_invalid_chess_cli_fails_before_authentication_or_controller(route, monkeypatch, arguments):
    forbid_keys(monkeypatch)
    assert cli.main(arguments) == 1
    route.assert_not_called()


def test_gemini_chess_voice_and_prompt_require_no_actual_capture(route, tmp_path):
    assert cli.main(CHESS + ["--voice-commands", "--command-audio", str(tmp_path / "missing.wav"), "--prompt-key"]) == 0
    assert route.call_args.kwargs["voice_commands"] is True
    cli.prompted_gemini_key.assert_called_once_with()
    cli.saved_gemini_key.assert_not_called()


def test_chess_interrupt_exit_code_is_preserved(route):
    route.return_value = SimpleNamespace(interrupted=True)
    assert cli.main(CHESS) == 130


@pytest.mark.parametrize("voice", [False, True])
def test_controller_forwards_exact_live_interface(monkeypatch, voice):
    module = ModuleType("vidi.core.chess_live")
    result = object()
    module.run_chess_live = Mock(return_value=result)
    monkeypatch.setitem(sys.modules, "vidi.core.chess_live", module)
    monkeypatch.setenv("DISPLAY", ":test")
    settings = AppSettings(simulation=SimulationSettings(preview=True))
    assert controller.run_simulation(settings, chess_live=True, continuous=True, speak=True, voice_commands=voice) is result
    module.run_chess_live.assert_called_once_with(settings, speak=True, voice_commands=voice, open_source=controller._open_source)


def test_invalid_direct_controller_chess_fails_before_import_or_camera(monkeypatch):
    monkeypatch.setattr(controller, "_open_source", lambda _: pytest.fail("Camera opened"))
    with pytest.raises(ValueError, match="chess-live requires"):
        controller.run_simulation(AppSettings(), chess_live=True)


def test_help_and_default_configuration_do_not_import_optional_chess(route, monkeypatch, capsys):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "chess" or name.startswith("chess."):
            pytest.fail("Optional chess dependency imported by a basic flow")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "--chess-live" in capsys.readouterr().out
    assert cli.main([]) == 0
    assert "chess_live" not in route.call_args.kwargs
