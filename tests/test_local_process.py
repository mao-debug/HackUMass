"""Local process boundaries without audio, network calls, or installed engines."""

import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from vidi.core import local_process as processes
from vidi.core.local_process import LocalProcessError, run_local_process


@pytest.fixture
def fake_process(monkeypatch):
    child = Mock(pid=12345, returncode=0)
    child.communicate.return_value = (b"output", b"warning")
    child.wait.return_value = 0
    popen = Mock(return_value=child)
    kill_group = Mock()
    monkeypatch.setattr(processes.subprocess, "Popen", popen)
    monkeypatch.setattr(processes.os, "killpg", kill_group)
    return child, popen, kill_group


def test_uses_literal_arguments_pipes_and_a_deadline_covering_input(monkeypatch, fake_process):
    child, popen, kill_group = fake_process
    monkeypatch.setattr(processes, "time", SimpleNamespace(monotonic=Mock(side_effect=[10, 10.25])))
    argv = ["local-engine", "--stdin", "literal ; $(command)"]

    result = run_local_process(argv, input_bytes=b"private input", timeout_s=2)

    assert result.args == argv
    assert result.returncode == 0
    assert result.stdout == b"output" and result.stderr == b"warning"
    arguments = popen.call_args.args[0]
    assert arguments == argv and arguments is not argv
    assert popen.call_args.kwargs == {
        "stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        "shell": False, "start_new_session": os.name == "posix", "close_fds": True,
    }
    child.communicate.assert_called_once_with(input=b"private input", timeout=1.75)
    kill_group.assert_not_called()
    for pipe in (child.stdin, child.stdout, child.stderr):
        pipe.close.assert_called_once_with()


def test_nonzero_exit_is_returned_for_engine_specific_handling(fake_process):
    child, _, _ = fake_process
    child.returncode = 7
    child.communicate.return_value = (b"partial text", b"engine error")

    result = run_local_process(["engine"], input_bytes=b"", timeout_s=1)

    assert result.returncode == 7
    assert result.stdout == b"partial text" and result.stderr == b"engine error"


@pytest.mark.parametrize("argv", [None, (), [], [""], ["engine", None], ["engine\x00bad"]])
def test_invalid_arguments_never_start_process(fake_process, argv):
    _, popen, _ = fake_process
    with pytest.raises(LocalProcessError, match="Invalid arguments"):
        run_local_process(argv, input_bytes=b"", timeout_s=1)
    popen.assert_not_called()


def test_input_must_be_bytes(fake_process):
    _, popen, _ = fake_process
    with pytest.raises(LocalProcessError, match="Invalid arguments"):
        run_local_process(["engine"], input_bytes="text", timeout_s=1)
    popen.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 10**500, "1"])
def test_invalid_deadlines_never_start_process(fake_process, timeout):
    _, popen, _ = fake_process
    with pytest.raises(LocalProcessError, match="finite positive"):
        run_local_process(["engine"], input_bytes=b"", timeout_s=timeout)
    popen.assert_not_called()


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError, OSError])
def test_start_errors_do_not_expose_raw_diagnostics(fake_process, failure, capsys):
    _, popen, _ = fake_process
    popen.side_effect = failure("PRIVATE command details")

    with pytest.raises(LocalProcessError) as caught:
        run_local_process(["engine"], input_bytes=b"PRIVATE input", timeout_s=1)

    assert "PRIVATE" not in str(caught.value)
    assert capsys.readouterr() == ("", "")


def test_timeout_terminates_then_kills_group_and_reaps_child(fake_process):
    child, _, kill_group = fake_process
    child.communicate.side_effect = subprocess.TimeoutExpired("engine", 1, output=b"PRIVATE output")
    child.wait.side_effect = [subprocess.TimeoutExpired("engine", 0.5), -signal.SIGKILL]

    with pytest.raises(LocalProcessError, match="timed out") as caught:
        run_local_process(["engine"], input_bytes=b"PRIVATE input", timeout_s=1)

    assert "PRIVATE" not in str(caught.value)
    assert kill_group.call_args_list == [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)]
    assert child.wait.call_args_list == [call(timeout=0.5), call(timeout=0.5)]
    for pipe in (child.stdin, child.stdout, child.stderr):
        pipe.close.assert_called_once_with()


def test_ctrl_c_propagates_after_group_cleanup(fake_process):
    child, _, kill_group = fake_process
    child.communicate.side_effect = KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_local_process(["engine"], input_bytes=b"", timeout_s=1)

    assert kill_group.call_args_list == [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)]
    assert child.wait.call_count == 2
    child.stdout.close.assert_called_once_with()


def test_group_is_killed_even_if_leader_exits_after_termination(fake_process):
    child, _, kill_group = fake_process
    child.communicate.side_effect = subprocess.TimeoutExpired("engine", 1)
    child.wait.return_value = 0

    with pytest.raises(LocalProcessError):
        run_local_process(["engine"], input_bytes=b"", timeout_s=1)

    kill_group.assert_has_calls([call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)])


def test_second_interrupt_during_cleanup_still_kills_and_propagates(fake_process):
    child, _, kill_group = fake_process
    child.communicate.side_effect = subprocess.TimeoutExpired("engine", 1)
    child.wait.side_effect = [KeyboardInterrupt, 0]

    with pytest.raises(KeyboardInterrupt):
        run_local_process(["engine"], input_bytes=b"", timeout_s=1)

    kill_group.assert_any_call(child.pid, signal.SIGKILL)
    assert child.wait.call_count == 2


@pytest.mark.parametrize("stage", ["terminate", "kill", "reap"])
def test_repeat_interrupt_during_signal_or_reap_cannot_skip_cleanup(fake_process, stage):
    child, _, kill_group = fake_process
    child.communicate.side_effect = subprocess.TimeoutExpired("engine", 1)
    previous_handler = signal.getsignal(signal.SIGINT)
    if stage == "terminate":
        kill_group.side_effect = [KeyboardInterrupt, None]
    elif stage == "kill":
        kill_group.side_effect = [None, KeyboardInterrupt, None]
    else:
        child.wait.side_effect = [subprocess.TimeoutExpired("engine", 0.5), KeyboardInterrupt, 0]

    with pytest.raises(KeyboardInterrupt):
        run_local_process(["engine"], input_bytes=b"", timeout_s=1)

    assert kill_group.call_args_list[-1] == call(child.pid, signal.SIGKILL)
    assert child.wait.call_count >= 2
    assert signal.getsignal(signal.SIGINT) == previous_handler


def test_communication_error_is_sanitized_and_cleans_up(fake_process):
    child, _, kill_group = fake_process
    child.communicate.side_effect = OSError("PRIVATE input details")

    with pytest.raises(LocalProcessError, match="could not complete") as caught:
        run_local_process(["engine"], input_bytes=b"", timeout_s=1)

    assert "PRIVATE" not in str(caught.value)
    kill_group.assert_any_call(child.pid, signal.SIGKILL)


def test_startup_time_consumes_the_same_deadline(monkeypatch, fake_process):
    child, _, kill_group = fake_process
    monkeypatch.setattr(processes, "time", SimpleNamespace(monotonic=Mock(side_effect=[1, 3])))

    with pytest.raises(LocalProcessError, match="timed out"):
        run_local_process(["engine"], input_bytes=b"", timeout_s=1)

    child.communicate.assert_not_called()
    kill_group.assert_any_call(child.pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="Mac/Linux process group lifecycle check")
def test_real_local_child_ignoring_termination_is_killed_and_reaped(monkeypatch, capsys):
    original_popen = subprocess.Popen
    children = []

    def record_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(processes.subprocess, "Popen", record_child)
    started = time.monotonic()
    with pytest.raises(LocalProcessError, match="timed out"):
        run_local_process(
            [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('captured only',flush=True); time.sleep(60)"],
            input_bytes=b"", timeout_s=0.3,
        )

    assert len(children) == 1
    assert children[0].returncode is not None and children[0].returncode < 0
    assert children[0].poll() is not None
    assert time.monotonic() - started < 4
    assert capsys.readouterr() == ("", "")


@pytest.mark.skipif(os.name != "posix", reason="Mac/Linux process group lifecycle check")
def test_real_local_child_is_reaped_when_communication_is_interrupted(monkeypatch):
    original_popen = subprocess.Popen
    children = []

    def interrupt_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        child.communicate = Mock(side_effect=KeyboardInterrupt)
        return child

    monkeypatch.setattr(processes.subprocess, "Popen", interrupt_child)

    with pytest.raises(KeyboardInterrupt):
        run_local_process([sys.executable, "-c", "import time; time.sleep(60)"], input_bytes=b"", timeout_s=2)

    assert len(children) == 1
    assert children[0].returncode is not None and children[0].returncode < 0
    assert children[0].poll() is not None
