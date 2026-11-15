"""The local diagnostic never records files, reads keys, or calls a service."""

from dataclasses import replace
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from vidi.audio.speech_to_text import SpeechRecognitionError
from vidi.audio.wake_word import WakeWordError
from vidi.core.configuration import AppSettings


@pytest.fixture
def diagnostic(monkeypatch, tmp_path):
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_wake.py"))["main"]
    namespace = main.__globals__
    settings = AppSettings()
    settings = replace(settings, assistant=replace(settings.assistant,
                       wake_model_dir_path=tmp_path, wake_keywords_path=tmp_path / "keywords.txt"))
    state = SimpleNamespace(opened=False, closed=False, released=False, prepared=False,
                            failure=None, dropped=0, no_audio=False)

    class Detector:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            if state.failure:
                raise WakeWordError("Invalid local model.")
            state.prepared = True

        def accept(self, data):
            return True

        def request_close(self):
            state.released = True

    class Input:
        def __init__(self, device):
            self.dropped_bytes = state.dropped

        def start(self):
            assert state.prepared
            state.opened = True

        def read(self, **kwargs):
            return b"" if state.no_audio else b"\x00\x40" * 320

        def close(self):
            state.closed = True

    clock = iter([0., .1, 2.])
    monkeypatch.setitem(namespace, "load_settings", lambda _: settings)
    monkeypatch.setitem(namespace, "WakeDetector", Detector)
    monkeypatch.setitem(namespace, "AudioInput", Input)
    monkeypatch.setitem(namespace, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    return main, state, tmp_path


def test_local_monitor_reports_only_counts_and_levels(diagnostic, capsys):
    main, state, directory = diagnostic
    assert main(["--seconds", "1"]) == 0
    output = capsys.readouterr().out
    assert '"wake_count": 1' in output
    assert '"peak_percent": 50.0' in output
    assert '"cloud_requests": 0' in output and '"audio_saved": false' in output
    assert state.closed and state.released
    assert list(directory.iterdir()) == []


def test_invalid_model_never_opens_microphone(diagnostic):
    main, state, _ = diagnostic
    state.failure = True
    assert main([]) == 1
    assert not state.opened and state.released


def test_audio_overrun_fails_closed(diagnostic, capsys):
    main, state, _ = diagnostic
    state.dropped = 640
    assert main(["--seconds", "1"]) == 1
    assert "dropped input" in capsys.readouterr().err
    assert state.closed


def test_no_audio_is_not_reported_as_a_working_microphone(diagnostic):
    main, state, _ = diagnostic
    state.no_audio = True
    assert main(["--seconds", "1"]) == 1
    assert state.closed and state.released


@pytest.mark.parametrize("seconds", ["0", "31", "-1"])
def test_unbounded_or_invalid_window_is_rejected_before_devices(diagnostic, seconds):
    main, state, _ = diagnostic
    with pytest.raises(SystemExit):
        main(["--seconds", seconds])
    assert not state.opened
