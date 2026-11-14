"""Configuration checks that do not need hardware, models, or network access."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from visionbridge.core.configuration import (
    AppSettings,
    CameraSettings,
    ConfigurationError,
    ModelSettings,
    SimulationSettings,
    load_settings,
)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_hardware_free_and_bounded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert isinstance(settings, AppSettings)
    assert settings.camera == CameraSettings()
    assert settings.camera.source == "sample"
    assert settings.camera.image_path is None
    assert settings.camera.webcam_index == 0
    assert (settings.camera.width, settings.camera.height) == (640, 480)
    assert settings.camera.fps == 5.0
    assert settings.simulation == SimulationSettings()
    assert settings.simulation.max_frames == 1
    assert settings.simulation.preview is False
    assert settings.simulation.output_path is None
    assert settings.models == ModelSettings()
    assert settings.log_level == "INFO"


@pytest.mark.parametrize(
    ("settings", "field", "replacement"),
    [
        (CameraSettings(), "source", "webcam"),
        (SimulationSettings(), "max_frames", 2),
        (ModelSettings(), "ocr_backend", "tesseract"),
        (AppSettings(), "log_level", "DEBUG"),
    ],
)
def test_settings_are_immutable(settings, field, replacement):
    with pytest.raises(FrozenInstanceError):
        setattr(settings, field, replacement)


def test_loads_explicit_values_and_resolves_relative_paths(tmp_path, monkeypatch):
    path = write_config(
        tmp_path,
        """\
camera:
  source: image
  image_path: samples/../sample.png
  webcam_index: 2
  width: 320
  height: 240
  fps: 2.5
simulation:
  max_frames: 3
  preview: true
  output_path: results/frame.png
models:
  object_detection_backend: configurable-detector
  object_detection_path: models/detector.onnx
  scene_description_backend: local
  ocr_backend: tesseract
  speech_recognition_backend: local
  text_to_speech_backend: local
log_level: DEBUG
""",
    )
    other_directory = tmp_path / "different-working-directory"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)

    settings = load_settings(path)

    assert settings.camera.source == "image"
    assert settings.camera.image_path == (tmp_path / "sample.png").resolve()
    assert settings.camera.webcam_index == 2
    assert (settings.camera.width, settings.camera.height) == (320, 240)
    assert settings.camera.fps == 2.5
    assert settings.simulation.max_frames == 3
    assert settings.simulation.preview is True
    assert settings.simulation.output_path == (tmp_path / "results/frame.png").resolve()
    assert settings.models.object_detection_backend == "configurable-detector"
    assert settings.models.object_detection_path == (tmp_path / "models/detector.onnx").resolve()
    assert settings.models.scene_description_backend == "local"
    assert settings.models.ocr_backend == "tesseract"
    assert settings.models.speech_recognition_backend == "local"
    assert settings.models.text_to_speech_backend == "local"
    assert settings.log_level == "DEBUG"


def test_absolute_paths_are_preserved(tmp_path):
    image = tmp_path / "sample.png"
    output = tmp_path / "output.png"
    model = tmp_path / "detector.onnx"
    path = write_config(
        tmp_path,
        f"camera:\n  source: image\n  image_path: {image}\n"
        f"simulation:\n  output_path: {output}\n"
        f"models:\n  object_detection_path: {model}\n",
    )

    settings = load_settings(path)

    assert settings.camera.image_path == image
    assert settings.simulation.output_path == output
    assert settings.models.object_detection_path == model


def test_partial_config_keeps_other_defaults(tmp_path):
    settings = load_settings(write_config(tmp_path, "camera:\n  width: 320\n"))

    assert settings.camera.width == 320
    assert settings.camera.height == 480
    assert settings.simulation == SimulationSettings()
    assert settings.models == ModelSettings()


def test_empty_mapping_uses_defaults(tmp_path):
    assert load_settings(write_config(tmp_path, "{}\n")) == AppSettings()


@pytest.mark.parametrize(
    "content",
    [
        "[]\n",
        "true\n",
        "42\n",
        "settings\n",
        "null\n",
        "camera: [\n",
        "unknown: true\n",
        "camera:\n  sourc: sample\n",
        "simulation:\n  frame_limit: 2\n",
        "models:\n  unsupported_backend: example\n",
        "camera: []\n",
        "simulation: false\n",
        "models: null\n",
        "camera:\n  source: picamera2\n",
        "camera:\n  source: image\n",
        "camera:\n  source: 1\n",
        "camera:\n  image_path: 123\n",
        "camera:\n  webcam_index: -1\n",
        "camera:\n  webcam_index: true\n",
        "camera:\n  webcam_index: 1.5\n",
        "camera:\n  width: 0\n",
        "camera:\n  width: true\n",
        "camera:\n  width: 1.5\n",
        "camera:\n  height: -1\n",
        "camera:\n  height: false\n",
        "camera:\n  fps: 0\n",
        "camera:\n  fps: -1\n",
        "camera:\n  fps: true\n",
        "camera:\n  fps: fast\n",
        "camera:\n  fps: .nan\n",
        "camera:\n  fps: .inf\n",
        "camera:\n  fps: -.inf\n",
        "simulation:\n  max_frames: 0\n",
        "simulation:\n  max_frames: -1\n",
        "simulation:\n  max_frames: true\n",
        "simulation:\n  max_frames: 1.5\n",
        'simulation:\n  preview: "true"\n',
        "simulation:\n  preview: 1\n",
        "simulation:\n  output_path: false\n",
        "models:\n  object_detection_backend: true\n",
        "models:\n  object_detection_path: 123\n",
        "models:\n  scene_description_backend: []\n",
        "models:\n  ocr_backend: 123\n",
        "models:\n  speech_recognition_backend: false\n",
        "models:\n  text_to_speech_backend: {}\n",
        "log_level: VERBOSE\n",
        "log_level: 10\n",
    ],
)
def test_invalid_configuration_reports_controlled_error(tmp_path, content):
    with pytest.raises(ConfigurationError):
        load_settings(write_config(tmp_path, content))


def test_missing_configuration_file_reports_controlled_error(tmp_path):
    with pytest.raises(ConfigurationError):
        load_settings(tmp_path / "missing.yaml")


def test_directory_as_configuration_file_reports_controlled_error(tmp_path):
    with pytest.raises(ConfigurationError):
        load_settings(tmp_path)


def test_non_utf8_configuration_reports_controlled_error(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ConfigurationError):
        load_settings(path)


def test_yaml_python_objects_are_not_constructed(tmp_path):
    path = write_config(tmp_path, "!!python/object:builtins.object {}\n")

    with pytest.raises(ConfigurationError):
        load_settings(path)
