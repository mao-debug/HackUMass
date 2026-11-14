"""Snapshot scene/proximity composition using fake local and cloud boundaries."""

from copy import deepcopy
from dataclasses import replace
import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vidi.ai import scene_description
from vidi.audio import text_to_speech
from vidi.core import controller
from vidi.core.configuration import (
    AppSettings, CameraSettings, ModelSettings, ProximitySettings, SimulationSettings,
    ConfigurationError, validate_runtime_settings,
)
from vidi.core.types import SpeechResult
from vidi.sensors import proximity as sensors


class Clock:
    now = 10.0

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(self.now * 1_000_000_000)

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def setup(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Snapshot tests cannot open a service, device, or audio engine")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(sensors, "ProximityMonitor", forbidden)
    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", forbidden)
    monkeypatch.setattr(text_to_speech, "LocalSpeech", forbidden)
    clock, events = Clock(), []
    monkeypatch.setattr(controller, "time", SimpleNamespace(
        monotonic=clock.monotonic, monotonic_ns=clock.monotonic_ns, sleep=lambda seconds: None,
    ))
    # refresh_snapshot uses its module clock unless an explicit now_ns is given.
    monkeypatch.setattr(sensors, "time", SimpleNamespace(monotonic_ns=clock.monotonic_ns))
    image = np.full((24, 32, 3), 70, np.uint8)

    class Camera:
        closed = False

        def __enter__(self):
            events.append("camera_enter")
            return self

        def read(self):
            events.append("capture")
            return image

        def __exit__(self, *args):
            self.closed = True
            events.append("camera_exit")

    camera = Camera()
    monkeypatch.setattr(controller, "_open_source", Mock(return_value=camera))
    settings = AppSettings(
        camera=CameraSettings(width=32, height=24),
        simulation=SimulationSettings(report_path=tmp_path / "snapshot.json"),
        models=ModelSettings(scene_description_backend="gemini", text_to_speech_backend="espeak_ng"),
        proximity=ProximitySettings(backend="fixture", fixture_path=tmp_path / "fixture.json"),
    )
    reports = []
    save = controller._save_report

    def checkpoint(path, report):
        reports.append(deepcopy(report))
        save(path, report)

    monkeypatch.setattr(controller, "_save_report", checkpoint)
    return SimpleNamespace(
        clock=clock, events=events, camera=camera, image=image, settings=settings, reports=reports,
    )


def install_cloud(monkeypatch, setup, *, failure=None):
    def describe(image, question):
        assert setup.camera.closed
        assert setup.reports[-1]["description_status"] == "pending"
        assert image is setup.image
        setup.events.append("cloud")
        setup.clock.advance(2.0)
        if failure is not None:
            raise failure
        return scene_description.SceneDescriptionResult(
            "A chair is beside a desk with a printed sign.", "test-scene-model", 2.0,
        )

    method = Mock(side_effect=describe)
    monkeypatch.setattr(scene_description, "GeminiSceneDescriber", Mock(return_value=SimpleNamespace(describe=method)))
    return method


def install_monitor(monkeypatch, setup, *, failure=None, close_delay=0):
    class Monitor:
        closed = False

        def __enter__(self):
            assert "cloud" in setup.events
            setup.events.append("sensor_enter")
            return self

        def fresh(self, *, after_ns, timeout_s):
            assert after_ns >= 12_000_000_000
            setup.events.append("fresh")
            setup.clock.advance(0.01)
            if failure is not None:
                raise failure
            sent = setup.clock.monotonic_ns()
            setup.clock.advance(0.01)
            received = setup.clock.monotonic_ns()
            return sensors.ProximitySnapshot(*(
                sensors.ProximityObservation(side, "ok", distance_mm=400,
                    request_sent_monotonic_ns=sent, received_monotonic_ns=received, reported_age_ms=0)
                for side in ("left", "right")
            ))

        def __exit__(self, *args):
            self.closed = True
            setup.events.append("sensor_exit")
            setup.clock.advance(close_delay)

    monitor = Monitor()
    monkeypatch.setattr(sensors, "ProximityMonitor", Mock(return_value=monitor))
    return monitor


def install_speech(monkeypatch, setup, callback=None):
    def speak(text, *, output_path):
        assert setup.events.index("camera_exit") < setup.events.index("cloud") < setup.events.index("fresh")
        assert setup.events[-1] == "sensor_exit"
        assert setup.reports[-1]["speech"]["status"] == "pending"
        assert "proximity" in setup.reports[-1]
        setup.events.append("speech")
        if callback is not None:
            return callback(text)
        return SpeechResult(status="completed", mode="playback", backend="fake", text=text)

    method = Mock(side_effect=speak)
    monkeypatch.setattr(text_to_speech, "LocalSpeech", Mock(return_value=SimpleNamespace(speak=method)))
    return method


def test_capture_closes_before_cloud_then_fresh_pair_and_full_checkpoint_before_audio(monkeypatch, setup):
    cloud = install_cloud(monkeypatch, setup)
    monitor = install_monitor(monkeypatch, setup)
    speech = install_speech(monkeypatch, setup)

    result = controller.run_simulation(setup.settings, describe=True, proximity=True, speak=True)

    assert setup.camera.closed and monitor.closed
    cloud.assert_called_once_with(setup.image, "What is in front of me?")
    utterance = speech.call_args.args[0]
    assert utterance.startswith("Left sensor: nearby return. Right sensor: nearby return.")
    assert "Camera snapshot from 2.0 seconds ago:" in utterance
    assert len(utterance.split()) <= setup.settings.speech.max_words
    report = json.loads(result.report_path.read_text())
    assert report["schema_version"] == 3 and report["mode"] == "phase4_scene_snapshot"
    assert report["description"] == "A chair is beside a desk with a printed sign."
    assert report["description_model"] == "test-scene-model"
    assert report["proximity_status"] == "ready" and report["speech"]["status"] == "completed"


def test_cloud_failure_still_allows_local_sensor_speech(monkeypatch, setup):
    install_cloud(monkeypatch, setup, failure=scene_description.SceneDescriptionError("Gemini timed out."))
    monitor = install_monitor(monkeypatch, setup)
    speech = install_speech(monkeypatch, setup)

    result = controller.run_simulation(setup.settings, describe=True, proximity=True, speak=True)

    assert result.description_status == "unavailable" and monitor.closed
    assert result.speech.status == "completed"
    assert "Scene description unavailable." in speech.call_args.args[0]
    assert "Camera snapshot" not in speech.call_args.args[0]


def test_sensor_failure_preserves_model_description_and_sanitizes_failure(monkeypatch, setup):
    install_cloud(monkeypatch, setup)
    monitor = install_monitor(monkeypatch, setup, failure=OSError("private serial diagnostics"))
    speech = install_speech(monkeypatch, setup)

    result = controller.run_simulation(setup.settings, describe=True, proximity=True, speak=True)

    assert monitor.closed and result.description_status == "ready"
    assert result.proximity_status == "unavailable"
    assert "reading unavailable" in speech.call_args.args[0]
    assert result.description in speech.call_args.args[0]
    assert "private serial diagnostics" not in result.report_path.read_text()


@pytest.mark.parametrize("stage", ["cloud", "sensors", "speech"])
def test_interruption_checkpoints_completed_channels_and_closes_resources(monkeypatch, setup, stage):
    install_cloud(monkeypatch, setup, failure=KeyboardInterrupt() if stage == "cloud" else None)
    monitor = None
    if stage != "cloud":
        monitor = install_monitor(monkeypatch, setup, failure=KeyboardInterrupt() if stage == "sensors" else None)
    if stage == "speech":
        install_speech(monkeypatch, setup, lambda text: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        controller.run_simulation(setup.settings, describe=True, proximity=True, speak=True)

    assert setup.camera.closed and (monitor is None or monitor.closed)
    report = json.loads(setup.settings.simulation.report_path.read_text())
    if stage == "cloud":
        assert report["description_status"] == "cancelled" and report["proximity"] is None
    else:
        assert report["description_status"] == "ready"
        assert report["proximity_status"] == ("cancelled" if stage == "sensors" else "ready")
    if stage == "speech":
        assert report["speech"]["status"] == "cancelled"


def test_sensor_close_delay_downgrades_exact_pair_before_speech(monkeypatch, setup):
    install_cloud(monkeypatch, setup)
    install_monitor(monkeypatch, setup, close_delay=0.6)
    speech = install_speech(monkeypatch, setup)

    result = controller.run_simulation(setup.settings, describe=True, proximity=True, speak=True)

    assert result.proximity_status == "unavailable"
    assert "nearby return" not in speech.call_args.args[0]
    assert result.proximity["left"]["status"] == "stale"


@pytest.mark.parametrize("delay_after_checkpoint", [3, 4])
def test_report_delay_cannot_make_old_sensor_return_sound_current(monkeypatch, setup, delay_after_checkpoint):
    install_cloud(monkeypatch, setup)
    install_monitor(monkeypatch, setup)
    speech = install_speech(monkeypatch, setup)
    save = controller._save_report
    calls = 0

    def delayed(path, report):
        nonlocal calls
        calls += 1
        save(path, report)
        if calls == delay_after_checkpoint:
            setup.clock.advance(0.6)

    monkeypatch.setattr(controller, "_save_report", delayed)
    result = controller.run_simulation(setup.settings, describe=True, proximity=True, speak=True)

    assert "nearby return" not in speech.call_args.args[0]
    assert result.proximity["left"]["status"] == "stale"
    assert result.proximity_status == "unavailable"


def test_configured_sensors_do_not_change_old_description_report_without_optin(monkeypatch, setup):
    install_cloud(monkeypatch, setup)

    result = controller.run_simulation(setup.settings, describe=True)

    report = json.loads(result.report_path.read_text())
    assert report["schema_version"] == 1
    assert not any(key.startswith("proximity") for key in report)
    assert result.description_status == "ready"


@pytest.mark.parametrize("destination", ["output", "report", "speech"])
def test_requested_exports_cannot_overwrite_proximity_fixture(setup, tmp_path, destination):
    fixture = tmp_path / "sensor-fixture.wav"
    config = replace(setup.settings, proximity=replace(setup.settings.proximity, fixture_path=fixture))
    if destination != "speech":
        config = replace(config, simulation=replace(config.simulation, **{f"{destination}_path": fixture}))

    with pytest.raises(ConfigurationError, match="fixture"):
        validate_runtime_settings(config, describe=True, proximity=True,
                                  speech_output=fixture if destination == "speech" else None)


@pytest.mark.parametrize("destination", ["output", "report"])
def test_scene_live_exports_cannot_overwrite_source_image(setup, tmp_path, destination):
    image = tmp_path / "source.png"
    config = replace(
        setup.settings,
        camera=replace(setup.settings.camera, source="image", image_path=image),
        simulation=replace(setup.settings.simulation, preview=True, **{f"{destination}_path": image}),
    )

    with pytest.raises(ConfigurationError, match="input image"):
        validate_runtime_settings(config, continuous=True, scene_live=True)
