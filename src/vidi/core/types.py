"""Portable records shared by hardware, fixtures, and reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A model class and confidence with an output-image pixel box (xyxy)."""

    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    """Only results associated with this frame; no temporal reuse."""

    frame_id: int
    backend: str
    status: str
    detections: tuple[Detection, ...] = ()
    simulated: bool = False


@dataclass(frozen=True)
class FramePacket:
    """An owned BGR image paired with its optional inference result."""

    frame_id: int
    captured_monotonic_ns: int
    image_bgr: np.ndarray
    inference: DetectionResult | None = None
