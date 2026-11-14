"""Validated YAML settings without hardware imports or service connections."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """A configuration file or command-line setting is invalid."""


def _integer(name: str, value: Any, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}.")


def _optional_text(name: str, value: Any) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigurationError(f"{name} must be a nonempty string or null.")


def _optional_path(name: str, value: Any) -> None:
    if value is not None and not isinstance(value, Path):
        raise ConfigurationError(f"{name} must be a pathlib.Path or null.")


@dataclass(frozen=True)
class CameraSettings:
    """Input settings; webcam is a development simulation source."""

    source: str = "sample"
    image_path: Path | None = None
    webcam_index: int = 0
    width: int = 640
    height: int = 480
    fps: float = 5.0

    def __post_init__(self) -> None:
        if self.source not in ("sample", "image", "webcam"):
            raise ConfigurationError("camera.source must be sample, image, or webcam.")
        _optional_path("camera.image_path", self.image_path)
        if self.source == "image" and self.image_path is None:
            raise ConfigurationError("camera.image_path is required for source=image.")
        _integer("camera.webcam_index", self.webcam_index, 0)
        _integer("camera.width", self.width)
        _integer("camera.height", self.height)
        if (
            type(self.fps) not in (int, float)
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ConfigurationError("camera.fps must be a finite number > 0.")


@dataclass(frozen=True)
class SimulationSettings:
    """A bounded frame-processing run with optional local display/export."""

    max_frames: int = 1
    preview: bool = False
    output_path: Path | None = None

    def __post_init__(self) -> None:
        _integer("simulation.max_frames", self.max_frames)
        if type(self.preview) is not bool:
            raise ConfigurationError("simulation.preview must be true or false.")
        _optional_path("simulation.output_path", self.output_path)


@dataclass(frozen=True)
class ModelSettings:
    """Reserved configuration only; later phases will select approved backends."""

    object_detection_backend: str | None = None
    object_detection_path: Path | None = None
    scene_description_backend: str | None = None
    ocr_backend: str | None = None
    speech_recognition_backend: str | None = None
    text_to_speech_backend: str | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("_path"):
                _optional_path(f"models.{item.name}", value)
            else:
                _optional_text(f"models.{item.name}", value)


@dataclass(frozen=True)
class AppSettings:
    """Complete local configuration, independent of the working directory."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigurationError("log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")


def _mapping(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a YAML mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown {name} setting(s): {', '.join(sorted(map(str, unknown)))}.")
    return dict(value)


def _section(document: dict[str, Any], name: str, cls: type, base: Path) -> Any:
    values = _mapping(document.get(name, {}), name, {item.name for item in fields(cls)})
    for key, value in values.items():
        if key.endswith("_path") and value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name}.{key} must be a nonempty path string or null.")
            path = Path(value).expanduser()
            values[key] = (base / path).resolve() if not path.is_absolute() else path.resolve()
    return cls(**values)


def load_settings(path: str | Path | None = None) -> AppSettings:
    """Load a safe YAML file, resolving file paths relative to its directory.

    With no path, use built-in defaults. Unknown keys fail instead of silently
    ignoring spelling mistakes. Environment files and credentials are not read.
    """
    if path is None:
        return AppSettings()
    config_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read configuration {config_path}: {exc}") from exc
    if document is None:
        raise ConfigurationError("Configuration must contain a YAML mapping, not an empty document.")
    values = _mapping(document, "root", {"camera", "simulation", "models", "log_level"})
    return AppSettings(
        camera=_section(values, "camera", CameraSettings, config_path.parent),
        simulation=_section(values, "simulation", SimulationSettings, config_path.parent),
        models=_section(values, "models", ModelSettings, config_path.parent),
        log_level=values.get("log_level", "INFO"),
    )
