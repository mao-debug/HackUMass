"""Validated YAML settings without hardware imports or service connections."""

from __future__ import annotations

import math
import os
import re
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
    if value is not None and (not isinstance(value, str) or not value.strip() or "\x00" in value):
        raise ConfigurationError(f"{name} must be a nonempty string or null.")


def _optional_path(name: str, value: Any) -> None:
    if value is not None and not isinstance(value, Path):
        raise ConfigurationError(f"{name} must be a pathlib.Path or null.")


@dataclass(frozen=True)
class CameraSettings:
    """Portable inputs and explicitly selected Raspberry Pi camera adapters."""

    source: str = "sample"
    image_path: Path | None = None
    webcam_index: int = 0
    width: int = 640
    height: int = 480
    fps: float = 5.0
    startup_timeout_s: float = 300.0
    capture_timeout_s: float = 5.0
    settle_s: float = 0.0
    autofocus: str = "off"
    autofocus_timeout_s: float = 4.0

    def __post_init__(self) -> None:
        if self.source not in ("sample", "image", "webcam", "picamera2", "imx500"):
            raise ConfigurationError("camera.source must be sample, image, webcam, picamera2, or imx500.")
        _optional_path("camera.image_path", self.image_path)
        if self.source == "image" and self.image_path is None:
            raise ConfigurationError("camera.image_path is required for source=image.")
        _integer("camera.webcam_index", self.webcam_index, 0)
        _integer("camera.width", self.width)
        _integer("camera.height", self.height)
        _positive_number("camera.fps", self.fps)
        _positive_number("camera.startup_timeout_s", self.startup_timeout_s)
        _positive_number("camera.capture_timeout_s", self.capture_timeout_s)
        if self.settle_s != 0 or type(self.settle_s) not in (int, float):
            _positive_number("camera.settle_s", self.settle_s)
        if self.autofocus not in ("off", "single_if_supported"):
            raise ConfigurationError("camera.autofocus must be off or single_if_supported.")
        _positive_number("camera.autofocus_timeout_s", self.autofocus_timeout_s)
        preparation_s = self.settle_s + (self.autofocus_timeout_s if self.autofocus != "off" else 0)
        if preparation_s >= self.startup_timeout_s:
            raise ConfigurationError("Camera preparation must fit within camera.startup_timeout_s.")


@dataclass(frozen=True)
class SimulationSettings:
    """Bounded runs and explicitly selected continuous preview/export options."""

    max_frames: int = 1
    preview: bool = False
    output_path: Path | None = None
    report_path: Path | None = None
    async_detection: bool = False

    def __post_init__(self) -> None:
        _integer("simulation.max_frames", self.max_frames)
        if type(self.preview) is not bool:
            raise ConfigurationError("simulation.preview must be true or false.")
        if type(self.async_detection) is not bool:
            raise ConfigurationError("simulation.async_detection must be true or false.")
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
    detection_max_output_tokens: int = 1024
    live_interval_s: float = 5.0
    ocr_max_output_tokens: int = 2048

    def __post_init__(self) -> None:
        _optional_text("cloud.model", self.model)
        if self.model is None:
            raise ConfigurationError("cloud.model must be a nonempty model identifier.")
        _positive_number("cloud.timeout_s", self.timeout_s)
        _positive_number("cloud.live_interval_s", self.live_interval_s)
        if self.live_interval_s < 5:
            raise ConfigurationError("cloud.live_interval_s must be at least 5 seconds.")
        _integer("cloud.max_output_tokens", self.max_output_tokens)
        _integer("cloud.max_image_edge", self.max_image_edge)
        _integer("cloud.detection_max_output_tokens", self.detection_max_output_tokens)
        _integer("cloud.ocr_max_output_tokens", self.ocr_max_output_tokens)
        if not 256 <= self.ocr_max_output_tokens <= 4096:
            raise ConfigurationError("cloud.ocr_max_output_tokens must be between 256 and 4096.")
        if not 128 <= self.detection_max_output_tokens <= 2048:
            raise ConfigurationError("cloud.detection_max_output_tokens must be between 128 and 2048.")
        if self.max_image_edge > 1024:
            raise ConfigurationError("cloud.max_image_edge cannot exceed the demo limit of 1024.")
        if self.max_output_tokens > 1024:
            raise ConfigurationError("cloud.max_output_tokens cannot exceed the demo limit of 1024.")


@dataclass(frozen=True)
class OcrSettings:
    """Printed-text recognition; configuration never enables reading or uploads."""

    language: str = "eng"
    psm: int = 6
    preprocess: str = "grayscale"
    timeout_s: float = 15.0
    executable: str = "tesseract"

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\+[A-Za-z][A-Za-z0-9_]*)*", self.language):
            raise ConfigurationError("ocr.language must be a language code such as eng or eng+spa.")
        if type(self.psm) is not int or self.psm not in (3, 6, 7, 11):
            raise ConfigurationError("ocr.psm must be 3, 6, 7, or 11.")
        if self.preprocess not in ("grayscale", "none"):
            raise ConfigurationError("ocr.preprocess must be grayscale or none.")
        _positive_number("ocr.timeout_s", self.timeout_s)
        _optional_text("ocr.executable", self.executable)
        if self.executable is None:
            raise ConfigurationError("ocr.executable must name the local Tesseract executable.")


@dataclass(frozen=True)
class ProximitySettings:
    """Optional local sensing; selecting a backend never opens a device."""

    backend: str = "disabled"
    port: str | None = None
    fixture_path: Path | None = None
    baud_rate: int = 115200
    poll_hz: float = 5.0
    response_timeout_s: float = 0.4
    stale_after_s: float = 0.5
    nearby_mm: int = 1000
    startup_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        if self.backend not in ("disabled", "fixture", "serial"):
            raise ConfigurationError("proximity.backend must be disabled, fixture, or serial.")
        _optional_text("proximity.port", self.port)
        _optional_path("proximity.fixture_path", self.fixture_path)
        if type(self.baud_rate) is not int or self.baud_rate != 115200:
            raise ConfigurationError("proximity.baud_rate must be 115200 for protocol version 1.")
        _positive_number("proximity.poll_hz", self.poll_hz)
        if self.poll_hz > 5:
            raise ConfigurationError("proximity.poll_hz cannot exceed 5 paired requests per second.")
        _positive_number("proximity.response_timeout_s", self.response_timeout_s)
        _positive_number("proximity.stale_after_s", self.stale_after_s)
        if not self.response_timeout_s <= self.stale_after_s <= 0.5:
            raise ConfigurationError("Proximity response timeout must fit inside a freshness limit of at most 0.5 seconds.")
        _integer("proximity.nearby_mm", self.nearby_mm, 20)
        if self.nearby_mm > 3000:
            raise ConfigurationError("proximity.nearby_mm cannot exceed the firmware's 3000 mm limit.")
        _positive_number("proximity.startup_timeout_s", self.startup_timeout_s)
        if self.startup_timeout_s > 10:
            raise ConfigurationError("proximity.startup_timeout_s cannot exceed 10 seconds.")


@dataclass(frozen=True)
class SpeechSettings:
    """One bounded utterance or explicit synthesis export, without queues."""

    speed_wpm: int = 150
    max_words: int = 60
    timeout_s: float = 45.0
    voice: str | None = None
    piper_python_path: Path | None = None
    piper_model_path: Path | None = None
    piper_length_scale: float = 1.0
    player: str | None = None

    def __post_init__(self) -> None:
        _integer("speech.speed_wpm", self.speed_wpm)
        _integer("speech.max_words", self.max_words)
        _positive_number("speech.timeout_s", self.timeout_s)
        _optional_text("speech.voice", self.voice)
        _optional_path("speech.piper_python_path", self.piper_python_path)
        _optional_path("speech.piper_model_path", self.piper_model_path)
        _positive_number("speech.piper_length_scale", self.piper_length_scale)
        if not 0.5 <= self.piper_length_scale <= 2.0:
            raise ConfigurationError("speech.piper_length_scale must be between 0.5 and 2.0.")
        if self.player not in (None, "paplay", "pw-play", "afplay", "aplay"):
            raise ConfigurationError("speech.player must be paplay, pw-play, afplay, aplay, or null for automatic selection.")


@dataclass(frozen=True)
class VoiceSettings:
    """One explicitly requested short recording; settings never open a microphone."""

    device: str = "pulse"
    record_seconds: int = 5
    timeout_s: float = 30.0
    model: str | None = None
    audio_path: Path | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.device, str) or not self.device.strip()
                or any(character in self.device for character in "\x00\r\n")):
            raise ConfigurationError("voice.device must be a nonempty literal input-device name without NUL or newlines.")
        _integer("voice.record_seconds", self.record_seconds)
        if self.record_seconds > 5:
            raise ConfigurationError("voice.record_seconds cannot exceed 5 seconds.")
        _positive_number("voice.timeout_s", self.timeout_s)
        if self.timeout_s > 60:
            raise ConfigurationError("voice.timeout_s cannot exceed 60 seconds.")
        _optional_text("voice.model", self.model)
        _optional_path("voice.audio_path", self.audio_path)


@dataclass(frozen=True)
class ChessSettings:
    """Requested board observation settings; no rules engine or camera import."""

    backend: str = "gemini"
    fixture_path: Path | None = None
    orientation: str = "white_near"
    observation_max_age_s: float = 30.0

    def __post_init__(self) -> None:
        if self.backend not in ("gemini", "fixture"):
            raise ConfigurationError("chess.backend must be gemini or fixture.")
        _optional_path("chess.fixture_path", self.fixture_path)
        if self.orientation not in ("white_near", "black_near"):
            raise ConfigurationError("chess.orientation must be white_near or black_near.")
        _positive_number("chess.observation_max_age_s", self.observation_max_age_s)
        if self.observation_max_age_s > 120:
            raise ConfigurationError("chess.observation_max_age_s cannot exceed 120 seconds.")


@dataclass(frozen=True)
class ModelSettings:
    """Local detector selection, optional descriptions, and future backends."""

    object_detection_backend: str | None = None
    object_detection_path: Path | None = None
    object_detection_config_path: Path | None = None
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
    ocr: OcrSettings = field(default_factory=OcrSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    proximity: ProximitySettings = field(default_factory=ProximitySettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    chess: ChessSettings = field(default_factory=ChessSettings)

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
            absolute = base / path if not path.is_absolute() else path
            # Preserve a venv interpreter's symlink so Python finds pyvenv.cfg.
            values[key] = Path(os.path.abspath(absolute)) if key == "piper_python_path" else absolute.resolve()
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
    values = _mapping(document, "root", {"camera", "simulation", "models", "log_level", "detection", "cloud", "ocr", "speech", "proximity", "voice", "chess"})
    return AppSettings(
        camera=_section(values, "camera", CameraSettings, config_path.parent),
        simulation=_section(values, "simulation", SimulationSettings, config_path.parent),
        models=_section(values, "models", ModelSettings, config_path.parent),
        log_level=values.get("log_level", "INFO"),
        detection=_section(values, "detection", DetectionSettings, config_path.parent),
        cloud=_section(values, "cloud", CloudSettings, config_path.parent),
        ocr=_section(values, "ocr", OcrSettings, config_path.parent),
        speech=_section(values, "speech", SpeechSettings, config_path.parent),
        proximity=_section(values, "proximity", ProximitySettings, config_path.parent),
        voice=_section(values, "voice", VoiceSettings, config_path.parent),
        chess=_section(values, "chess", ChessSettings, config_path.parent),
    )


def validate_runtime_settings(
    settings: AppSettings, *, describe: bool = False, read_text: bool = False,
    speak: bool = False, speech_output: Path | None = None, gemini_detect: bool = False,
    continuous: bool = False,
    gemini_live: bool = False,
    gemini_read: bool = False,
    scene_live: bool = False,
    proximity: bool = False,
    voice_commands: bool = False,
    offline: bool = False,
    chess_live: bool = False,
) -> None:
    """Check backend combinations after command-line overrides are applied."""
    backend = settings.models.object_detection_backend or "none"
    if chess_live:
        if not continuous or not settings.simulation.preview:
            raise ConfigurationError("--chess-live requires --continuous --preview for keyboard activation.")
        if any((scene_live, describe, read_text, gemini_detect, gemini_live, gemini_read)):
            raise ConfigurationError("Choose --chess-live by itself as the camera action.")
        if backend != "none" or settings.camera.source == "imx500":
            raise ConfigurationError("Chess preview requires detector none and an ordinary camera source.")
        if proximity:
            raise ConfigurationError("Chess mode does not support --proximity; use the general scene mode for sensors.")
        if speech_output is not None:
            raise ConfigurationError("Chess mode supports requested playback, not --speech-output WAV export.")
        if settings.chess.backend == "fixture":
            if settings.chess.fixture_path is None:
                raise ConfigurationError("Fixture chess requires --chess-fixture or chess.fixture_path.")
            if voice_commands:
                raise ConfigurationError("Fixture chess cannot enable cloud voice commands or microphone capture.")
        elif offline:
            raise ConfigurationError("--offline chess requires the fixture backend; Gemini board observation needs explicit cloud access.")
    if offline:
        if not (scene_live or chess_live) or not continuous or not settings.simulation.preview:
            raise ConfigurationError("--offline requires --scene-live --continuous --preview.")
        if voice_commands:
            raise ConfigurationError("--offline cannot enable voice commands or microphone capture.")
    if voice_commands:
        if not (scene_live or chess_live) or not continuous or not settings.simulation.preview:
            raise ConfigurationError("--voice-commands requires --scene-live --continuous --preview for explicit M activation.")
        if settings.models.speech_recognition_backend not in (None, "gemini"):
            raise ConfigurationError("Voice commands support the existing gemini speech-recognition backend.")
        if not chess_live and settings.models.ocr_backend not in (None, "gemini"):
            raise ConfigurationError("Voice reading requires the gemini OCR backend.")
        if settings.voice.audio_path is not None and settings.voice.audio_path.suffix.lower() != ".wav":
            raise ConfigurationError("voice.audio_path and --command-audio require a local .wav input.")
    if scene_live:
        if not continuous or not settings.simulation.preview:
            raise ConfigurationError("--scene-live requires --continuous and --preview for keyboard activation.")
        if any((describe, read_text, gemini_detect, gemini_live, gemini_read)):
            raise ConfigurationError("Choose --scene-live by itself as the camera scene action.")
        if backend != "none" or settings.camera.source == "imx500":
            raise ConfigurationError("Scene preview requires detector none and an ordinary camera source.")
        if speech_output is not None:
            raise ConfigurationError("Live scene mode supports requested playback, not WAV export.")
        if settings.models.scene_description_backend not in (None, "gemini"):
            raise ConfigurationError("Scene preview supports the existing gemini scene backend.")
    if proximity:
        if not (describe or scene_live):
            raise ConfigurationError("--proximity requires --describe or --continuous --scene-live.")
        if settings.proximity.backend == "disabled":
            raise ConfigurationError("Select a fixture or serial proximity backend before requesting sensing.")
        if settings.proximity.backend == "serial" and settings.proximity.port is None:
            raise ConfigurationError("Serial sensing requires --sensor-port or proximity.port; use scripts/check_sensors.py --list.")
        if settings.proximity.backend == "fixture" and settings.proximity.fixture_path is None:
            raise ConfigurationError("Fixture sensing requires proximity.fixture_path.")
    if gemini_read:
        if read_text or describe or gemini_detect or gemini_live or continuous:
            raise ConfigurationError("Choose --gemini-read by itself as the reading action; it cannot be combined with local reading or other cloud/live actions.")
        if backend != "none":
            raise ConfigurationError("Gemini reading requires --detector none or the pi-gemini-ocr profile.")
        if settings.models.ocr_backend not in (None, "gemini"):
            raise ConfigurationError("--gemini-read requires the gemini OCR backend.")
        if settings.camera.source == "imx500":
            raise ConfigurationError("Use --source picamera2 for a Raspberry Pi Gemini reading snapshot.")
    if gemini_live and not continuous:
        raise ConfigurationError("--gemini-live requires --continuous and authorizes repeated camera images to Google.")
    if continuous and not (scene_live or chess_live) and (describe or read_text or gemini_detect or speak or speech_output is not None):
        raise ConfigurationError("Continuous mode cannot be combined with snapshot reading, descriptions, detection, or speech actions.")
    if settings.simulation.async_detection and not continuous:
        raise ConfigurationError("The asynchronous live profile requires --continuous.")
    if settings.simulation.async_detection and backend not in ("none", "opencv_ssd", "fixture", "gemini"):
        raise ConfigurationError("Asynchronous live preview supports ordinary capture with local or Gemini detection.")
    if (speak or speech_output is not None) and not (read_text or gemini_read or describe or gemini_detect or scene_live or chess_live):
        raise ConfigurationError("Speech requires --read-text, --gemini-read, --describe, or --gemini-detect.")
    if speak and speech_output is not None:
        raise ConfigurationError("Choose either --speak or --speech-output, not both.")
    if read_text:
        if describe or gemini_detect:
            raise ConfigurationError("--read-text cannot be combined with a Gemini action.")
        if backend != "none":
            raise ConfigurationError("Reading mode requires --detector none or the pi-ocr profile.")
        if settings.models.ocr_backend not in (None, "tesseract"):
            raise ConfigurationError("--read-text supports local tesseract OCR; use --gemini-read to explicitly send an image to Gemini.")
    if backend == "gemini" and not (gemini_detect or gemini_live):
        raise ConfigurationError("Gemini detection requires --gemini-detect for one image or --continuous --gemini-live for repeated images.")
    if gemini_detect or gemini_live:
        if backend != "gemini":
            raise ConfigurationError("Gemini detection actions require the gemini detector.")
        if describe:
            raise ConfigurationError("Choose --gemini-detect or --describe for one Gemini request, not both.")
        if settings.camera.source == "imx500":
            raise ConfigurationError("Use --source picamera2 for a Raspberry Pi Gemini snapshot.")
        if settings.detection.max_detections > 20:
            raise ConfigurationError("Gemini snapshots support at most 20 objects per request.")
        if settings.detection.labels_path is not None or settings.detection.fixture_path is not None:
            raise ConfigurationError("Gemini uses its returned labels; remove local labels and fixture paths.")
    if speak or speech_output is not None:
        if settings.models.text_to_speech_backend not in (None, "espeak_ng", "say", "piper"):
            raise ConfigurationError("Speech backend must be espeak_ng, say, or piper.")
        if speech_output is not None:
            if not isinstance(speech_output, Path) or speech_output.suffix.lower() != ".wav":
                raise ConfigurationError("--speech-output requires a local .wav path.")
            if settings.models.text_to_speech_backend == "say":
                raise ConfigurationError("WAV export requires espeak_ng or piper; say supports playback only.")
    if backend not in ("none", "imx500", "opencv_ssd", "fixture", "gemini"):
        raise ConfigurationError("Object detector must be none, imx500, opencv_ssd, fixture, or gemini.")
    if backend == "opencv_ssd":
        if settings.camera.source == "imx500":
            raise ConfigurationError("Use --source picamera2 for CPU detection on a Raspberry Pi camera.")
        if settings.models.object_detection_path is None or settings.models.object_detection_config_path is None:
            raise ConfigurationError("The opencv_ssd detector requires local --model and --model-config paths.")
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
    if (read_text or gemini_read or gemini_detect or gemini_live or describe or scene_live or chess_live) and settings.camera.image_path is not None:
        for other in (output, report):
            if other is not None and other.resolve() == settings.camera.image_path.resolve():
                raise ConfigurationError("Exports must not overwrite the input image.")
    if chess_live and settings.chess.fixture_path is not None:
        for other in (output, report, speech_output):
            if other is not None and other.resolve() == settings.chess.fixture_path.resolve():
                raise ConfigurationError("Exports must not overwrite the chess fixture input.")
    if proximity and settings.proximity.backend == "fixture" and settings.proximity.fixture_path is not None:
        for other in (output, report, speech_output):
            if other is not None and other.resolve() == settings.proximity.fixture_path.resolve():
                raise ConfigurationError("Exports must not overwrite the proximity fixture input.")
    if (voice_commands or chess_live) and settings.voice.audio_path is not None:
        for other in (output, report, speech_output):
            if other is not None and other.resolve() == settings.voice.audio_path.resolve():
                raise ConfigurationError("Exports must not overwrite the voice audio input.")
    if speech_output is not None:
        for other in (output, report, settings.camera.image_path):
            if other is not None and speech_output.resolve() == other.resolve():
                raise ConfigurationError("The WAV, image input/output, and JSON report must use different paths.")
