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


def _positive_number(name: str, value: Any) -> None:
    if type(value) not in (int, float):
        raise ConfigurationError(f"{name} must be a finite number > 0.")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ConfigurationError(f"{name} must be a finite number > 0.")


def _optional_text(name: str, value: Any) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigurationError(f"{name} must be a nonempty string or null.")


def _optional_path(name: str, value: Any) -> None:
    if value is not None and not isinstance(value, Path):
        raise ConfigurationError(f"{name} must be a pathlib.Path or null.")


@dataclass(frozen=True)
class CameraSettings:
    """Portable inputs plus the explicitly selected Pi AI Camera adapter."""

    source: str = "sample"
    image_path: Path | None = None
    webcam_index: int = 0
    width: int = 640
    height: int = 480
    fps: float = 5.0
    startup_timeout_s: float = 300.0
    capture_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.source not in ("sample", "image", "webcam", "imx500"):
            raise ConfigurationError("camera.source must be sample, image, webcam, or imx500.")
        _optional_path("camera.image_path", self.image_path)
        if self.source == "image" and self.image_path is None:
            raise ConfigurationError("camera.image_path is required for source=image.")
        _integer("camera.webcam_index", self.webcam_index, 0)
        _integer("camera.width", self.width)
        _integer("camera.height", self.height)
        _positive_number("camera.fps", self.fps)
        _positive_number("camera.startup_timeout_s", self.startup_timeout_s)
        _positive_number("camera.capture_timeout_s", self.capture_timeout_s)


@dataclass(frozen=True)
class SimulationSettings:
    """A bounded frame-processing run with optional local display/export."""

    max_frames: int = 1
    preview: bool = False
    output_path: Path | None = None
    report_path: Path | None = None

    def __post_init__(self) -> None:
        _integer("simulation.max_frames", self.max_frames)
        if type(self.preview) is not bool:
            raise ConfigurationError("simulation.preview must be true or false.")
        _optional_path("simulation.output_path", self.output_path)
        _optional_path("simulation.report_path", self.report_path)


@dataclass(frozen=True)
class DetectionSettings:
    """Shared filtering and explicit simulation assets, without downloads."""

    confidence_threshold: float = 0.55
    max_detections: int = 10
    labels_path: Path | None = None
    fixture_path: Path | None = None

    def __post_init__(self) -> None:
        threshold = self.confidence_threshold
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ConfigurationError("detection.confidence_threshold must be a number between 0 and 1.")
        _integer("detection.max_detections", self.max_detections)
        _optional_path("detection.labels_path", self.labels_path)
        _optional_path("detection.fixture_path", self.fixture_path)


@dataclass(frozen=True)
class CloudSettings:
    """Optional Gemini request limits; settings alone never enable uploads."""

    model: str = "gemini-3.5-flash-lite"
    timeout_s: float = 15.0
    max_output_tokens: int = 256
    max_image_edge: int = 1024

    def __post_init__(self) -> None:
        _optional_text("cloud.model", self.model)
        if self.model is None:
            raise ConfigurationError("cloud.model must be a nonempty model identifier.")
        _positive_number("cloud.timeout_s", self.timeout_s)
        _integer("cloud.max_output_tokens", self.max_output_tokens)
        _integer("cloud.max_image_edge", self.max_image_edge)
        if self.max_image_edge > 1024:
            raise ConfigurationError("cloud.max_image_edge cannot exceed the demo limit of 1024.")
        if self.max_output_tokens > 1024:
            raise ConfigurationError("cloud.max_output_tokens cannot exceed the demo limit of 1024.")


@dataclass(frozen=True)
class ModelSettings:
    """Local detector selection, optional descriptions, and future backends."""

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
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    cloud: CloudSettings = field(default_factory=CloudSettings)

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
    values = _mapping(document, "root", {"camera", "simulation", "models", "log_level", "detection", "cloud"})
    return AppSettings(
        camera=_section(values, "camera", CameraSettings, config_path.parent),
        simulation=_section(values, "simulation", SimulationSettings, config_path.parent),
        models=_section(values, "models", ModelSettings, config_path.parent),
        log_level=values.get("log_level", "INFO"),
        detection=_section(values, "detection", DetectionSettings, config_path.parent),
        cloud=_section(values, "cloud", CloudSettings, config_path.parent),
    )


def validate_runtime_settings(settings: AppSettings, *, describe: bool = False) -> None:
    """Check backend combinations after command-line overrides are applied."""
    backend = settings.models.object_detection_backend or "none"
    if backend not in ("none", "imx500", "fixture"):
        raise ConfigurationError("Object detector must be none, imx500, or fixture.")
    if backend == "imx500":
        if settings.camera.source != "imx500":
            raise ConfigurationError("The imx500 detector requires --source imx500.")
        if settings.models.object_detection_path is None:
            raise ConfigurationError("The imx500 detector requires a local model path (--model).")
    if backend == "fixture":
        if settings.camera.source not in ("sample", "image"):
            raise ConfigurationError("Fixture detections require a sample or image source, never a live camera.")
        if settings.detection.fixture_path is None:
            raise ConfigurationError("The fixture detector requires --detections-fixture or detection.fixture_path.")
    if describe and settings.models.scene_description_backend not in (None, "gemini"):
        raise ConfigurationError("This demo supports only the gemini scene description backend.")
    output = settings.simulation.output_path
    report = settings.simulation.report_path
    if output is not None and report is not None and output.resolve() == report.resolve():
        raise ConfigurationError("The image output and JSON report must use different paths.")
