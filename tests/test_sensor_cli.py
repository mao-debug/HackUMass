"""CLI routing is tested with camera/controller/credentials boundaries mocked."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vidi.core import controller
from vidi.core.configuration import AppSettings, ModelSettings, ProximitySettings, SimulationSettings
from vidi.core.credentials import CredentialError
import vidi.main as cli
from vidi.sensors import proximity


@pytest.fixture
def routes(monkeypatch):
    result = SimpleNamespace(interrupted=False, ready_frames=1, description_status="ready",
                             speech=SimpleNamespace(status="not_requested"))
    run = Mock(return_value=result)
    monkeypatch.setattr(controller, "run_simulation", run)
    monkeypatch.setattr(proximity, "create_source", lambda _: pytest.fail("CLI opened sensor"))
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: nullcontext())
    return run


def configured(monkeypatch, *, backend="serial", port="/dev/serial/by-id/my-board", fixture_path=None):
    settings = AppSettings(simulation=SimulationSettings(preview=True),
                           proximity=ProximitySettings(backend=backend, port=port, fixture_path=fixture_path))
    monkeypatch.setattr(cli, "load_settings", lambda _: settings)
    return settings


@pytest.mark.parametrize("flags", [
    ["--scene-live"], ["--scene-live", "--continuous"],
    ["--scene-live", "--preview"],
    ["--scene-live", "--continuous", "--preview", "--describe"],
    ["--scene-live", "--continuous", "--preview", "--read-text"],
    ["--scene-live", "--continuous", "--preview", "--gemini-read"],
    ["--scene-live", "--continuous", "--preview", "--gemini-live"],
    ["--scene-live", "--continuous", "--preview", "--gemini-detect"],
    ["--scene-live", "--continuous", "--preview", "--detector", "fixture"],
    ["--scene-live", "--continuous", "--preview", "--speech-output", "result.wav"],
    ["--sensor-port", "/explicit"],
    ["--describe", "--sensor-port", "/explicit"],
])
def test_invalid_flags_fail_before_credentials_or_controller(routes, monkeypatch, flags):
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: pytest.fail("Credentials accessed"))
    monkeypatch.setattr(cli, "prompted_gemini_key", lambda: pytest.fail("Prompt opened"))
    assert cli.main(flags) == 1
    routes.assert_not_called()


def test_scene_live_routes_only_explicit_requested_actions(routes):
    assert cli.main(["--scene-live", "--continuous", "--preview"]) == 0
    assert routes.call_args.kwargs == dict(continuous=True, scene_live=True, proximity=False,
                                          speak=False, question="What is in front of me?")


def test_scene_live_proximity_port_override_is_literal_and_reaches_controller(routes, monkeypatch):
    configured(monkeypatch, port=None)
    assert cli.main(["--scene-live", "--continuous", "--proximity", "--sensor-port",
                     "/dev/serial/by-id/exact", "--speak", "--question", "Describe the room."]) == 0
    assert routes.call_args.args[0].proximity.port == "/dev/serial/by-id/exact"
    assert routes.call_args.kwargs == dict(continuous=True, scene_live=True, proximity=True,
                                          speak=True, question="Describe the room.")


def test_snapshot_proximity_routes_with_description_and_optional_speech(routes, monkeypatch):
    configured(monkeypatch)
    assert cli.main(["--describe", "--proximity", "--speak"]) == 0
    assert routes.call_args.kwargs == dict(describe=True, proximity=True, speak=True,
                                          speech_output=None, question="What is in front of me?")


@pytest.mark.parametrize("flags", [["--proximity"], ["--read-text", "--proximity"],
                                   ["--continuous", "--proximity"]])
def test_proximity_without_scene_action_never_reads_a_source(routes, monkeypatch, flags):
    configured(monkeypatch)
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: pytest.fail("Credentials accessed"))
    assert cli.main(flags) == 1
    routes.assert_not_called()


@pytest.mark.parametrize("backend", ["disabled", "serial", "fixture"])
def test_proximity_missing_source_details_fails_before_key_access(routes, monkeypatch, backend):
    configured(monkeypatch, backend=backend, port=None)
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: pytest.fail("Credentials accessed"))
    assert cli.main(["--describe", "--proximity"]) == 1
    routes.assert_not_called()


def test_configuration_alone_does_not_enable_sensing_or_cloud_credentials(routes, monkeypatch):
    base = configured(monkeypatch)
    settings = replace(base, models=ModelSettings(scene_description_backend="gemini"))
    monkeypatch.setattr(cli, "load_settings", lambda _: settings)
    monkeypatch.setattr(cli, "saved_gemini_key", lambda: pytest.fail("Credentials accessed"))
    assert cli.main([]) == 0
    assert routes.call_args.kwargs == {}


def test_unreadable_saved_key_does_not_disable_local_scene_preview(routes, monkeypatch, caplog):
    configured(monkeypatch)

    @contextmanager
    def bad_saved_key():
        raise CredentialError("Saved key permissions are invalid.")
        yield

    monkeypatch.setattr(cli, "saved_gemini_key", bad_saved_key)
    assert cli.main(["--scene-live", "--continuous", "--proximity"]) == 0
    assert routes.call_args.kwargs["proximity"] is True
    assert "Local proximity remains available" in caplog.text


def test_explicit_prompt_failure_still_stops_before_scene_preview(routes, monkeypatch):
    configured(monkeypatch)

    @contextmanager
    def failed_prompt():
        raise CredentialError("A secure terminal is required.")
        yield

    monkeypatch.setattr(cli, "prompted_gemini_key", failed_prompt)
    assert cli.main(["--scene-live", "--continuous", "--proximity", "--prompt-key"]) == 1
    routes.assert_not_called()


def test_scene_live_reports_interruption_with_existing_exit_code(routes):
    routes.return_value.interrupted = True
    assert cli.main(["--scene-live", "--continuous", "--preview"]) == 130
