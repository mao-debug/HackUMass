"""Private key prompting stays explicit and never prints a key."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vidi.core import controller, credentials
from vidi.main import main


@pytest.fixture
def hidden_prompt(monkeypatch):
    reader = Mock(return_value="PRIVATE-FAKE-KEY")
    monkeypatch.setattr(credentials.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(credentials.getpass, "getpass", reader)
    return reader


@pytest.mark.parametrize("arguments", [["--gemini-detect"], ["--describe"]])
def test_prompt_scopes_key_to_requested_cloud_run(monkeypatch, hidden_prompt, arguments, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "previous-value")

    def simulate(settings, **kwargs):
        assert os.environ["GEMINI_API_KEY"] == "PRIVATE-FAKE-KEY"
        return SimpleNamespace(
            last_inference=SimpleNamespace(status="ready"), ready_frames=1,
            description_status="ready", speech=SimpleNamespace(status="not_requested"),
        )

    monkeypatch.setattr(controller, "run_simulation", simulate)
    assert main(arguments + ["--prompt-key"]) == 0
    hidden_prompt.assert_called_once()
    assert os.environ["GEMINI_API_KEY"] == "previous-value"
    captured = capsys.readouterr()
    assert "PRIVATE-FAKE-KEY" not in captured.out + captured.err


def test_default_run_never_prompts(monkeypatch, hidden_prompt):
    assert main([]) == 0
    hidden_prompt.assert_not_called()


def test_prompt_is_rejected_without_cloud_action(monkeypatch, hidden_prompt):
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main(["--prompt-key"]) == 1
    hidden_prompt.assert_not_called()


def test_invalid_cloud_flags_are_rejected_before_prompt(hidden_prompt):
    assert main(["--prompt-key", "--gemini-detect", "--read-text"]) == 1
    hidden_prompt.assert_not_called()


@pytest.mark.parametrize("arguments", [
    ["--prompt-key", "PRIVATE-FAKE-KEY"],
    ["--prompt-key=PRIVATE-FAKE-KEY"],
    ["--prompt-k=PRIVATE-FAKE-KEY"],
])
def test_key_as_flag_argument_is_rejected_without_echoing_it(hidden_prompt, capsys, arguments):
    with pytest.raises(SystemExit) as error:
        main(["--gemini-detect"] + arguments)
    assert error.value.code == 2
    assert "PRIVATE-FAKE-KEY" not in capsys.readouterr().err
    hidden_prompt.assert_not_called()


def test_prompt_cancellation_stops_before_camera(monkeypatch, hidden_prompt):
    hidden_prompt.side_effect = KeyboardInterrupt
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(controller, "_open_source", lambda settings: pytest.fail("Camera opened"))
    assert main(["--gemini-detect", "--prompt-key"]) == 130
    assert "GEMINI_API_KEY" not in os.environ


def test_runtime_failure_restores_original_key(monkeypatch, hidden_prompt):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(controller, "run_simulation", Mock(side_effect=controller.SimulationError("Test failure")))
    assert main(["--gemini-detect", "--prompt-key"]) == 1
    assert "GEMINI_API_KEY" not in os.environ
