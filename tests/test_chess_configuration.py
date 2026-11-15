"""Chess settings and runtime gates do not load an observer or rules package."""

from dataclasses import replace
from pathlib import Path

import pytest

from vidi.core.configuration import (
    AppSettings, CameraSettings, ChessSettings, ConfigurationError, ModelSettings,
    SimulationSettings, VoiceSettings, load_settings, validate_runtime_settings,
)


def configured(**changes):
    return replace(AppSettings(simulation=SimulationSettings(preview=True)), **changes)


def validate(settings, **options):
    arguments = dict(chess_live=True, continuous=True)
    arguments.update(options)
    validate_runtime_settings(settings, **arguments)


def test_chess_defaults_are_reserved_until_explicit_mode():
    settings = load_settings()
    assert settings.chess == ChessSettings()
    assert settings.chess.backend == "gemini"
    assert settings.chess.fixture_path is None
    assert settings.chess.orientation == "white_near"
    assert settings.chess.observation_max_age_s == 30
    validate_runtime_settings(settings)


@pytest.mark.parametrize("field,value", [
    ("backend", "disabled"), ("backend", None), ("backend", []),
    ("orientation", "auto"), ("orientation", None),
    ("observation_max_age_s", 0), ("observation_max_age_s", -1),
    ("observation_max_age_s", True), ("observation_max_age_s", "30"),
    ("observation_max_age_s", float("nan")), ("observation_max_age_s", float("inf")),
    ("observation_max_age_s", 120.01), ("observation_max_age_s", 10**1000),
    ("fixture_path", "not-a-Path"),
])
def test_chess_setting_validation_is_strict(field, value):
    with pytest.raises(ConfigurationError, match="chess\\."):
        ChessSettings(**{field: value})


def test_age_upper_boundary_and_orientation_are_supported():
    assert ChessSettings(observation_max_age_s=120, orientation="black_near").observation_max_age_s == 120


def test_yaml_chess_paths_resolve_against_yaml_parent(tmp_path, monkeypatch):
    directory = tmp_path / "profiles"
    directory.mkdir()
    path = directory / "board.yaml"
    path.write_text("chess:\n  backend: fixture\n  fixture_path: ../boards/missing.json\n  orientation: black_near\n  observation_max_age_s: 12.5\n")
    monkeypatch.chdir(tmp_path)
    chess = load_settings(path).chess
    assert chess.fixture_path == tmp_path / "boards" / "missing.json"
    assert chess.orientation == "black_near" and chess.observation_max_age_s == 12.5
    assert not chess.fixture_path.exists()  # Parsing never reads the fixture.


@pytest.mark.parametrize("document", ["chess: []\n", "chess:\n  confidence: 0.9\n", "chess:\n  fixture_path: 12\n"])
def test_yaml_chess_unknown_keys_types_and_paths_fail(tmp_path, document):
    path = tmp_path / "invalid.yaml"
    path.write_text(document)
    with pytest.raises(ConfigurationError):
        load_settings(path)


@pytest.mark.parametrize("options", [
    {"continuous": False}, {"scene_live": True}, {"describe": True},
    {"read_text": True}, {"gemini_read": True}, {"gemini_detect": True},
    {"gemini_live": True}, {"proximity": True}, {"speech_output": Path("reading.wav")},
    {"offline": True},
])
def test_conflicting_runtime_actions_fail(options):
    with pytest.raises(ConfigurationError):
        validate(configured(), **options)


@pytest.mark.parametrize("settings", [
    configured(simulation=SimulationSettings(preview=False)),
    configured(models=ModelSettings(object_detection_backend="opencv_ssd")),
    configured(camera=CameraSettings(source="imx500")),
    configured(chess=ChessSettings(backend="fixture")),
])
def test_required_preview_detector_source_and_fixture(settings):
    with pytest.raises(ConfigurationError):
        validate(settings)


def test_gemini_chess_allows_explicit_voice_and_local_playback():
    validate(configured(), voice_commands=True, speak=True)


def test_fixture_is_offline_with_optional_local_speech_but_no_voice(tmp_path):
    settings = configured(chess=ChessSettings(backend="fixture", fixture_path=tmp_path / "board.json"))
    validate(settings, offline=True, speak=True)
    with pytest.raises(ConfigurationError, match="Fixture chess"):
        validate(settings, voice_commands=True)


@pytest.mark.parametrize("input_kind", ["fixture", "image", "audio"])
@pytest.mark.parametrize("output_kind", ["image", "report"])
def test_chess_exports_never_overwrite_inputs_even_without_voice(tmp_path, input_kind, output_kind):
    fixture = tmp_path / "board.json"
    image = tmp_path / "board.png"
    audio = tmp_path / "command.wav"
    target = {"fixture": fixture, "image": image, "audio": audio}[input_kind]
    simulation = SimulationSettings(preview=True, **{
        "output_path" if output_kind == "image" else "report_path": target,
    })
    settings = configured(
        camera=CameraSettings(source="image", image_path=image),
        chess=ChessSettings(backend="fixture", fixture_path=fixture),
        voice=VoiceSettings(audio_path=audio), simulation=simulation,
    )
    with pytest.raises(ConfigurationError, match="overwrite"):
        validate(settings)


def test_profiles_match_the_intended_sources_and_optional_modes():
    root = Path(__file__).resolve().parents[1]
    pi = load_settings(root / "config/pi-chess.yaml")
    simulation = load_settings(root / "config/chess-sim.yaml")
    assert (pi.camera.source, pi.camera.width, pi.camera.height, pi.camera.fps) == ("picamera2", 1280, 960, 15)
    assert pi.chess.backend == "gemini" and pi.speech.max_words == 100 and pi.speech.timeout_s == 60
    assert pi.cloud.timeout_s == 60 and pi.voice.record_seconds == 5
    assert simulation.camera.source == "sample" and simulation.chess.backend == "fixture"
    assert simulation.chess.fixture_path == root / "examples/chess-fixture.json"
    validate(replace(pi, simulation=replace(pi.simulation, preview=True)), voice_commands=True, speak=True)
    validate(replace(simulation, simulation=replace(simulation.simulation, preview=True)), offline=True)
