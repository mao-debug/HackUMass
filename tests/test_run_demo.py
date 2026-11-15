"""Demo launch plans and handoff are tested without starting the application."""

import builtins
import importlib.util
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SECRET = "offline-launcher-test-secret-not-a-real-key"


@pytest.fixture
def launcher(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_demo.py"
    name = "vidi_demo_launcher_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    # No test may accidentally replace pytest with a running camera application.
    monkeypatch.setattr(module.os, "execve", Mock(side_effect=AssertionError("Real exec is forbidden")))
    return module


def plan(launcher, root, *flags):
    return launcher.build_plan(launcher.build_parser().parse_args(flags), root=root)


def option(argv, name):
    return argv[argv.index(name) + 1]


def test_default_dry_run_is_offline_and_has_no_application_or_environment_side_effects(launcher, monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"vidi", "google", "cv2", "picamera2", "sounddevice", "serial"}
        return original_import(name, *args, **kwargs)

    environment = SimpleNamespace(
        get=Mock(side_effect=AssertionError("Credentials read during planning")),
        copy=Mock(side_effect=AssertionError("Environment copied during planning")),
    )
    # Replace only the script's reference, leaving pytest's own environment intact.
    monkeypatch.setattr(launcher, "os", SimpleNamespace(path=os.path, environ=environment))
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=AssertionError("Planning created a directory")))
    execute = Mock(side_effect=AssertionError("Dry run started an application"))
    monkeypatch.setattr(launcher, "_exec_plan", execute)

    assert launcher.main([]) == 0
    output = capsys.readouterr().out
    assert "synthetic simulation" in output and "disabled (offline)" in output
    assert "--offline" in output and "Plan only" in output
    execute.assert_not_called()
    environment.get.assert_not_called()
    environment.copy.assert_not_called()


def test_explicit_run_hands_only_the_validated_plan_to_executor(launcher, monkeypatch):
    execute = Mock()
    monkeypatch.setattr(launcher, "_exec_plan", execute)
    assert launcher.main(["--run", "--proximity"]) == 0
    execute.assert_called_once()
    selected = execute.call_args.args[0]
    assert selected.target == "sim" and not selected.cloud and not selected.voice
    assert "--offline" in selected.argv and "--proximity" in selected.argv


@pytest.mark.parametrize("target,environment,profile", [
    ("sim", ".venv", "phase5-sim.yaml"), ("pi", ".venv-pi", "pi-phase5.yaml"),
])
def test_target_selects_existing_environment_and_profile_without_installing(launcher, tmp_path, target, environment, profile):
    root = tmp_path / "project not created"
    selected = plan(launcher, root, "--target", target)
    assert selected.argv[0] == str(root / environment / "bin" / "python")
    assert selected.argv[1:3] == ("-m", "vidi")
    assert option(selected.argv, "--config") == str(root / "config" / profile)
    assert selected.root == root and not root.exists()
    assert all(flag in selected.argv for flag in ("--continuous", "--scene-live", "--preview", "--offline"))


def test_full_pi_plan_preserves_explicit_device_arguments_without_shell_interpretation(launcher, tmp_path):
    root = tmp_path / "repo with spaces"
    device = "/dev/serial/by-id/USB bridge $(never-run);literal"
    microphone = "plughw:CARD=USB headset,DEV=0"
    selected = plan(launcher, root, "--target", "pi", "--cloud", "--voice", "--speak", "--proximity",
                    "--sensor-port", device, "--microphone-device", microphone, "--prompt-key")
    assert selected.cloud and selected.voice and selected.target == "pi"
    assert "--offline" not in selected.argv
    assert "--voice-commands" in selected.argv and "--speak" in selected.argv and "--prompt-key" in selected.argv
    assert option(selected.argv, "--sensor-port") == device
    assert option(selected.argv, "--microphone-device") == microphone
    assert shlex.split(shlex.join(selected.argv)) == list(selected.argv)


def test_simulated_voice_requires_explicit_cloud_and_replay_wav(launcher, tmp_path):
    audio = tmp_path / "my spoken command.wav"
    selected = plan(launcher, tmp_path, "--cloud", "--voice", "--command-audio", str(audio))
    assert selected.cloud and selected.voice and "--voice-commands" in selected.argv
    assert option(selected.argv, "--command-audio") == str(audio)
    assert not audio.exists()  # Planning does not read or create it.


@pytest.mark.parametrize("flags", [
    ["--voice"], ["--prompt-key"], ["--sensor-port", "/dev/ttyUSB0"],
    ["--target", "pi", "--proximity"],
    ["--proximity", "--sensor-port", "/dev/ttyUSB0"],
    ["--command-audio", "command.wav"], ["--microphone-device", "pulse"],
    ["--cloud", "--voice"],
    ["--cloud", "--voice", "--command-audio", "command.mp3"],
    ["--target", "pi", "--cloud", "--voice", "--microphone-device", "pulse", "--command-audio", "command.wav"],
    ["--target", "pi", "--cloud", "--voice", "--microphone-device", "bad\ninput"],
    ["--target", "pi", "--proximity", "--sensor-port", "bad\x00port"],
])
def test_invalid_combinations_fail_before_executor(launcher, monkeypatch, flags):
    execute = Mock(side_effect=AssertionError("Invalid plan was executed"))
    monkeypatch.setattr(launcher, "_exec_plan", execute)
    with pytest.raises(SystemExit) as caught:
        launcher.main(["--run", *flags])
    assert caught.value.code == 2
    execute.assert_not_called()


def test_relative_overrides_are_resolved_from_calling_directory(launcher, tmp_path, monkeypatch):
    working = tmp_path / "working directory"
    working.mkdir()
    monkeypatch.chdir(working)
    root = tmp_path / "project"
    selected = plan(launcher, root, "--python", "tools/python", "--artifacts-dir", "my reports",
                    "--cloud", "--voice", "--command-audio", "clips/command.WAV")
    assert selected.argv[0] == str(working / "tools" / "python")
    assert selected.report_path.parent == working / "my reports"
    assert option(selected.argv, "--command-audio") == str(working / "clips" / "command.WAV")
    assert selected.root == root


def test_virtual_environment_interpreter_symlink_is_preserved(launcher, tmp_path):
    root = tmp_path / "project"
    binary = root / ".venv" / "bin" / "python"
    binary.parent.mkdir(parents=True)
    target = tmp_path / "base-python"
    target.write_text("never executed")
    binary.symlink_to(target)
    selected = plan(launcher, root)
    assert selected.argv[0] == str(binary)
    assert selected.argv[0] != str(binary.resolve())
    override = plan(launcher, root, "--python", str(binary))
    assert override.argv[0] == str(binary)


def test_each_plan_gets_a_distinct_json_report_without_creating_directory(launcher, tmp_path):
    first = plan(launcher, tmp_path)
    second = plan(launcher, tmp_path)
    assert first.report_path != second.report_path
    assert first.report_path.suffix == second.report_path.suffix == ".json"
    assert first.report_path.parent == second.report_path.parent == tmp_path / "artifacts"
    assert option(first.argv, "--report") == str(first.report_path)
    assert not first.report_path.parent.exists()


def test_executor_uses_execve_argument_list_and_private_environment_copy(launcher, tmp_path, monkeypatch, capsys):
    root = tmp_path / "project with spaces"
    root.mkdir()
    interpreter = tmp_path / "selected python"
    interpreter.write_text("not executed")
    interpreter.chmod(0o700)
    selected = plan(launcher, root, "--python", str(interpreter))
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("VIDI_EXISTING_SETTING", "preserved")
    monkeypatch.setenv("PYTHONPATH", "old search path")
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    execute = Mock()
    change_directory = Mock()
    monkeypatch.setattr(launcher.os, "execve", execute)
    monkeypatch.setattr(launcher.os, "chdir", change_directory)

    launcher._exec_plan(selected)

    execute.assert_called_once()
    executable, arguments, environment = execute.call_args.args
    assert executable == str(interpreter) and arguments == list(selected.argv)
    assert isinstance(arguments, list)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"] == str(root / "src")
    assert environment["GEMINI_API_KEY"] == SECRET
    assert environment["VIDI_EXISTING_SETTING"] == "preserved"
    assert os.environ["PYTHONNOUSERSITE"] == "0" and os.environ["PYTHONPATH"] == "old search path"
    change_directory.assert_called_once_with(root)
    assert selected.report_path.parent.is_dir()
    assert SECRET not in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["missing", "directory", "non_executable"])
def test_interpreter_preflight_fails_before_mkdir_chdir_or_exec(launcher, tmp_path, monkeypatch, kind):
    interpreter = tmp_path / "python"
    if kind == "directory":
        interpreter.mkdir()
    elif kind == "non_executable":
        interpreter.write_text("not executable")
        interpreter.chmod(0o600)
    selected = plan(launcher, tmp_path / "root", "--python", str(interpreter))
    change_directory = Mock()
    monkeypatch.setattr(launcher.os, "chdir", change_directory)
    with pytest.raises(OSError, match="interpreter is unavailable"):
        launcher._exec_plan(selected)
    launcher.os.execve.assert_not_called()
    change_directory.assert_not_called()
    assert not selected.report_path.parent.exists()


def test_launch_failure_returns_one_without_leaking_environment_or_raw_error(launcher, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setattr(launcher, "_exec_plan", Mock(side_effect=OSError(f"private failure {SECRET}")))
    assert launcher.main(["--run"]) == 1
    output = capsys.readouterr()
    assert "Launch failed" in output.err and "No installation was attempted" in output.err
    assert SECRET not in output.out + output.err and "private failure" not in output.err


def test_launcher_interruption_returns_130(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "_exec_plan", Mock(side_effect=KeyboardInterrupt))
    assert launcher.main(["--run"]) == 130


def test_foreground_handoff_does_not_replace_child_exit_status_with_launcher_success(launcher, monkeypatch):
    # A real exec replaces this process. SystemExit models the eventual child
    # status without ever launching another interpreter or the application.
    monkeypatch.setattr(launcher, "_exec_plan", Mock(side_effect=SystemExit(7)))
    with pytest.raises(SystemExit) as caught:
        launcher.main(["--run"])
    assert caught.value.code == 7


def test_displayed_command_keeps_paths_quoted_and_does_not_print_saved_key(launcher, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    interpreter = tmp_path / "interpreter with spaces"
    assert launcher.main(["--python", str(interpreter), "--cloud", "--prompt-key"]) == 0
    output = capsys.readouterr().out
    command_line = next(line.removeprefix("Command: ") for line in output.splitlines() if line.startswith("Command: "))
    words = shlex.split(command_line)
    assert words[0] == "PYTHONNOUSERSITE=1"
    assert words[2] == str(interpreter)
    assert "--prompt-key" in words and SECRET not in output
