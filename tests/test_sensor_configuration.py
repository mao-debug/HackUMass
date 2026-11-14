"""Phase 4 settings stay inert until explicit scene/sensor activation."""

from dataclasses import replace
from pathlib import Path

import pytest

from vidi.core.configuration import (
    AppSettings, CameraSettings, ConfigurationError, ModelSettings, ProximitySettings,
    SimulationSettings, load_settings, validate_runtime_settings,
)


def scene_settings(**changes):
    return replace(AppSettings(simulation=SimulationSettings(preview=True)), **changes)


def test_proximity_defaults_are_disabled_and_do_not_require_devices():
    settings = load_settings()
    assert settings.proximity == ProximitySettings()
    assert settings.proximity.backend == "disabled" and settings.proximity.port is None
    validate_runtime_settings(settings)


@pytest.mark.parametrize("values", [
    {"backend": "automatic"}, {"backend": None}, {"port": ""}, {"port": " \t"},
    {"port": Path("/dev/ttyUSB0")}, {"port": "bad\x00port"}, {"fixture_path": "pair.json"},
    {"baud_rate": 9600}, {"baud_rate": True}, {"poll_hz": 0}, {"poll_hz": 5.01},
    {"poll_hz": True}, {"poll_hz": float("nan")}, {"response_timeout_s": 0},
    {"response_timeout_s": .6}, {"response_timeout_s": .4, "stale_after_s": .3},
    {"stale_after_s": .51}, {"stale_after_s": float("inf")},
    {"nearby_mm": 19}, {"nearby_mm": 3001}, {"nearby_mm": True}, {"nearby_mm": 1000.0},
    {"startup_timeout_s": 0}, {"startup_timeout_s": 10.01}, {"startup_timeout_s": "3"},
])
def test_proximity_values_reject_unbounded_or_ambiguous_settings(values):
    with pytest.raises(ConfigurationError):
        ProximitySettings(**values)


def test_load_resolves_fixture_path_but_preserves_stable_serial_port(tmp_path):
    directory = tmp_path / "config"
    directory.mkdir()
    config = directory / "local.yaml"
    # The deliberately missing fixture is not read during configuration loading.
    config.write_text("proximity:\n  backend: serial\n  port: /dev/serial/by-id/my-board\n"
                      "  fixture_path: fixtures/missing.json\n")
    settings = load_settings(config)
    assert settings.proximity.port == "/dev/serial/by-id/my-board"
    assert settings.proximity.fixture_path == directory / "fixtures" / "missing.json"
    assert not settings.proximity.fixture_path.exists()


def test_loading_sensor_settings_does_not_import_or_open_serial(tmp_path, monkeypatch):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"serial", "picamera2", "libcamera", "google"}
        return original(name, *args, **kwargs)

    config = tmp_path / "serial.yaml"
    config.write_text("proximity:\n  backend: serial\n  port: /not/a/real/device\n")
    monkeypatch.setattr(builtins, "__import__", guarded)
    settings = load_settings(config)
    validate_runtime_settings(settings)


@pytest.mark.parametrize("document", [
    "proximity:\n  backend: fixture\n  typo_poll_hz: 5\n",
    "proximity:\n  fixture_path: 2\n", "proximity: [serial]\n",
])
def test_yaml_rejects_unknown_sensor_keys_and_wrong_types(tmp_path, document):
    config = tmp_path / "bad.yaml"
    config.write_text(document)
    with pytest.raises(ConfigurationError):
        load_settings(config)


@pytest.mark.parametrize("continuous,preview", [(False, False), (False, True), (True, False)])
def test_scene_live_requires_continuous_and_preview(continuous, preview):
    settings = scene_settings(simulation=SimulationSettings(preview=preview))
    with pytest.raises(ConfigurationError, match="continuous.*preview"):
        validate_runtime_settings(settings, scene_live=True, continuous=continuous)


@pytest.mark.parametrize("action", ["describe", "read_text", "gemini_detect", "gemini_live", "gemini_read"])
def test_scene_live_cannot_combine_with_other_cloud_or_reading_actions(action):
    with pytest.raises(ConfigurationError, match="by itself"):
        validate_runtime_settings(scene_settings(), scene_live=True, continuous=True, **{action: True})


@pytest.mark.parametrize("backend", ["fixture", "opencv_ssd", "gemini", "imx500"])
def test_scene_live_requires_detector_none(backend):
    settings = scene_settings(models=ModelSettings(object_detection_backend=backend))
    with pytest.raises(ConfigurationError, match="detector none"):
        validate_runtime_settings(settings, scene_live=True, continuous=True)


def test_scene_live_ordinary_camera_and_no_wav_export():
    settings = scene_settings(camera=CameraSettings(source="imx500"))
    with pytest.raises(ConfigurationError, match="ordinary camera"):
        validate_runtime_settings(settings, scene_live=True, continuous=True)
    with pytest.raises(ConfigurationError, match="WAV export"):
        validate_runtime_settings(scene_settings(), scene_live=True, continuous=True,
                                  speech_output=Path("speech.wav"))


def test_scene_live_without_proximity_remains_valid_without_a_sensor_port():
    validate_runtime_settings(scene_settings(), scene_live=True, continuous=True)
    validate_runtime_settings(scene_settings(), scene_live=True, continuous=True, speak=True)


@pytest.mark.parametrize("proximity,expected", [
    (ProximitySettings(), "fixture or serial"),
    (ProximitySettings(backend="serial"), "Serial sensing requires"),
    (ProximitySettings(backend="fixture"), "fixture_path"),
])
def test_explicit_proximity_needs_selected_source(proximity, expected):
    with pytest.raises(ConfigurationError, match=expected):
        validate_runtime_settings(scene_settings(proximity=proximity), describe=True, proximity=True)


def test_proximity_requires_a_description_action_and_never_autoselects_a_port():
    settings = scene_settings(proximity=ProximitySettings(backend="serial", port="/explicit"))
    with pytest.raises(ConfigurationError, match="requires --describe"):
        validate_runtime_settings(settings, proximity=True)
    validate_runtime_settings(settings, describe=True, proximity=True)
    validate_runtime_settings(settings, scene_live=True, continuous=True, proximity=True)


def test_new_phase4_profiles_load_without_opening_fixture_or_device():
    directory = Path(__file__).resolve().parents[1] / "config"
    pi = load_settings(directory / "pi-phase4.yaml")
    simulation = load_settings(directory / "phase4-sim.yaml")
    assert pi.camera.source == "picamera2"
    assert (pi.models.object_detection_backend or "none") == "none"
    assert pi.proximity.backend == "serial" and pi.proximity.port is None
    assert simulation.camera.source in {"sample", "image"}
    assert simulation.proximity.backend == "fixture" and simulation.proximity.fixture_path is not None
    assert (simulation.models.object_detection_backend or "none") == "none"
    for settings in (pi, simulation):
        # The preview remains an explicit CLI selection, even in a scene profile.
        settings = replace(settings, simulation=replace(settings.simulation, preview=True))
        validate_runtime_settings(settings, scene_live=True, continuous=True)
