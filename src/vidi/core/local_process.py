"""Bounded local command execution without a shell or inherited output streams."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import threading
import time


_TERMINATE_GRACE_S = 0.5
_KILL_GRACE_S = 0.5


class LocalProcessError(RuntimeError):
    """A sanitized process failure without command output or input contents."""


class LocalProcessCancelled(LocalProcessError):
    """An explicit caller cancellation, distinct from an engine failure."""


def _signal_process_group(process: subprocess.Popen[bytes], sig: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except (ProcessLookupError, OSError):
        # The process may have exited between the failure and cleanup.
        pass


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Stop descendants too, even when the direct child already exited."""
    interrupted = False
    previous_handler = None
    deferred_sigint = False

    def defer_interrupt(signum, frame) -> None:
        nonlocal interrupted
        interrupted = True

    try:
        # A second Ctrl+C must not interrupt the very cleanup required by the
        # first. Signal handlers are only configurable from the main thread;
        # explicit KeyboardInterrupt exceptions are guarded below as well.
        try:
            previous_handler = signal.signal(signal.SIGINT, defer_interrupt)
            deferred_sigint = True
        except (ValueError, OSError):
            pass
        try:
            _signal_process_group(process, signal.SIGTERM)
        except KeyboardInterrupt:
            interrupted = True
        try:
            process.wait(timeout=_TERMINATE_GRACE_S)
        except (subprocess.TimeoutExpired, OSError):
            pass
        except KeyboardInterrupt:
            interrupted = True
    finally:
        # A parent can exit after TERM while a playback helper ignores it.
        # Always clear the group instead of consulting only parent.poll().
        for _ in range(2):
            try:
                _signal_process_group(process, signal.SIGKILL)
                break
            except KeyboardInterrupt:
                interrupted = True
        for _ in range(2):
            try:
                process.wait(timeout=_KILL_GRACE_S)
                break
            except (subprocess.TimeoutExpired, OSError):
                break
            except KeyboardInterrupt:
                interrupted = True
        if deferred_sigint:
            signal.signal(signal.SIGINT, previous_handler)
    if interrupted:
        raise KeyboardInterrupt


def _close_pipes(process: subprocess.Popen[bytes]) -> None:
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


def run_local_process(
    argv: list[str], *, input_bytes: bytes, timeout_s: float,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run fixed arguments and communicate bytes within a finite deadline.

    Nonzero exits are returned to the caller for engine-specific handling.
    Startup/communication errors and timeouts produce sanitized errors. Ctrl+C
    propagates after cleanup. Commands run in a new POSIX session so cancellation
    can terminate both the engine and any playback helpers it created.
    """
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(argument, str) or "\x00" in argument for argument in argv)
        or not argv[0]
        or not isinstance(input_bytes, bytes)
    ):
        raise LocalProcessError("Invalid arguments for the requested local process.")
    try:
        valid_timeout = (
            type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise LocalProcessError("The local process timeout must be a finite positive number.")
    if cancel_event is not None and cancel_event.is_set():
        raise LocalProcessCancelled("The local process was cancelled.")

    arguments = argv.copy()
    deadline = time.monotonic() + timeout_s
    try:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name == "posix",
            close_fds=True,
        )
    except FileNotFoundError:
        raise LocalProcessError("The requested local executable is unavailable.") from None
    except OSError:
        raise LocalProcessError("The requested local executable could not be started.") from None

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(arguments, timeout_s)
        if cancel_event is None:
            stdout, stderr = process.communicate(input=input_bytes, timeout=remaining)
        else:
            first = True
            while True:
                if cancel_event.is_set():
                    raise LocalProcessCancelled("The local process was cancelled.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(arguments, timeout_s)
                try:
                    stdout, stderr = process.communicate(
                        input=input_bytes if first else None, timeout=min(0.05, remaining),
                    )
                    if cancel_event.is_set():
                        raise LocalProcessCancelled("The local process was cancelled.")
                    break
                except subprocess.TimeoutExpired:
                    first = False
        return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    except LocalProcessCancelled:
        _stop_process(process)
        raise
    except subprocess.TimeoutExpired:
        _stop_process(process)
        raise LocalProcessError("The local process timed out.") from None
    except KeyboardInterrupt:
        _stop_process(process)
        raise
    except Exception:
        _stop_process(process)
        raise LocalProcessError("The local process could not complete.") from None
    finally:
        _close_pipes(process)
