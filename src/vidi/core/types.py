"""Portable records shared by hardware, fixtures, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    """An image-space box; open-vocabulary cloud labels have no class ID/score."""

    class_id: int | None
    label: str
    confidence: float | None
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    """Only results associated with this frame; no temporal reuse."""

    frame_id: int
    backend: str
    status: str
    detections: tuple[Detection, ...] = ()
    simulated: bool = False
    model: str | None = None
    elapsed_s: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class FramePacket:
    """An owned BGR image paired with its optional inference result."""

    frame_id: int
    captured_monotonic_ns: int
    image_bgr: np.ndarray
    inference: DetectionResult | None = None


@dataclass(frozen=True)
class OcrResult:
    """One transcription attempt; cloud text is a model prediction to verify."""

    status: str = "not_requested"
    frame_id: int | None = None
    text: str = ""
    backend: str | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class SpeechResult:
    """Engine completion does not establish that physical audio was audible."""

    status: str = "not_requested"
    mode: str | None = None
    backend: str | None = None
    text: str = ""
    truncated: bool = False
    elapsed_s: float = 0.0
    output_path: Path | None = None
    error: str | None = None
