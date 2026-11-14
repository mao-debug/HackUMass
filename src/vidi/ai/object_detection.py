"""Validated local detections and explicitly simulated fixture playback.

Fixtures replay annotations; they never recognize objects. The hardware SSD
decoder lives with its IMX500 request adapter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np

from vidi.core.types import Detection, DetectionResult, FramePacket

if TYPE_CHECKING:
    from vidi.core.configuration import DetectionSettings

PLACEHOLDER_LABELS = {"", "-", "???", "n/a", "unknown", "undefined", "background", "__background__"}


class DetectionError(RuntimeError):
    """A local detector's assets or outputs are invalid or unsupported."""


def supported_label(label: str) -> bool:
    """Keep original class indices while excluding undefined label entries."""
    return label.strip().lower() not in PLACEHOLDER_LABELS


def load_labels(path: Path) -> tuple[str, ...]:
    """Read one label per class index, retaining blank and placeholder entries."""
    try:
        labels = tuple(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError) as exc:
        raise DetectionError(f"Cannot read labels file {path}: {exc}") from exc
    if not labels or any(len(label) > 256 for label in labels):
        raise DetectionError("Labels must contain class-indexed lines of at most 256 characters.")
    return labels


def filter_detections(
    detections: Iterable[Detection], settings: DetectionSettings
) -> tuple[Detection, ...]:
    """Apply confidence, supported-label filtering, and a deterministic limit."""
    valid = (
        item for item in detections
        if supported_label(item.label) and item.confidence >= settings.confidence_threshold
    )
    return tuple(sorted(valid, key=lambda item: -item.confidence)[:settings.max_detections])


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class FixtureDetector:
    """Replay dimension-matched annotations from an explicit local JSON fixture."""

    def __init__(self, path: Path, settings: DetectionSettings) -> None:
        self.settings = settings
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise DetectionError(f"Cannot read detection fixture {path}: {exc}") from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"version", "image_size", "detections"}
            or type(document["version"]) is not int
            or document["version"] != 1
        ):
            raise DetectionError("Detection fixture requires version 1, image_size, and detections only.")
        size = document["image_size"]
        if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise DetectionError("Fixture image_size must be [positive width, positive height].")
        self.image_size = tuple(size)
        raw = document["detections"]
        if not isinstance(raw, list):
            raise DetectionError("Fixture detections must be a list.")
        labels = load_labels(settings.labels_path) if settings.labels_path else None
        detections = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or set(item) != {"class_id", "label", "confidence", "bbox_xyxy"}:
                raise DetectionError(f"Invalid detection fixture entry {index}.")
            class_id, label = item["class_id"], item["label"]
            score, box = item["confidence"], item["bbox_xyxy"]
            if type(class_id) is not int or class_id < 0 or not isinstance(label, str) or len(label) > 256:
                raise DetectionError(f"Invalid class or label in fixture entry {index}.")
            if not _number(score) or not 0 <= score <= 1:
                raise DetectionError(f"Invalid confidence in fixture entry {index}.")
            if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box):
                raise DetectionError(f"Invalid bbox_xyxy in fixture entry {index}.")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
                raise DetectionError(f"Out-of-bounds or empty bbox_xyxy in fixture entry {index}.")
            if labels is not None:
                if class_id >= len(labels):
                    continue
                label = labels[class_id]
            detections.append(Detection(class_id, label.strip(), float(score), tuple(box)))
        self._detections = filter_detections(detections, settings)

    def detect(self, packet: FramePacket) -> DetectionResult:
        """Return simulated annotations for this frame ID, never rescale silently."""
        image = packet.image_bgr
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or (image.shape[1], image.shape[0]) != self.image_size:
            raise DetectionError(
                f"Fixture expects {self.image_size[0]}x{self.image_size[1]} BGR frames; "
                f"received shape {image.shape}."
            )
        return DetectionResult(packet.frame_id, "fixture", "ready", self._detections, simulated=True)


def render_detections(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Annotate a copy and visibly identify every fixture result as simulated."""
    annotated = image.copy()
    for item in result.detections:
        x0, y0, x1, y1 = item.bbox_xyxy
        cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (60, 220, 80), 2)
        cv2.putText(
            annotated, f"{item.label} {item.confidence:.2f}", (x0, max(12, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 80), 1, cv2.LINE_AA,
        )
    if result.simulated:
        message = "SIMULATED DETECTIONS"
    elif result.status == "not_ready":
        message = "INFERENCE NOT READY"
    else:
        message = "OBJECT DETECTION - PROTOTYPE"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(
        annotated, message, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
        min(0.6, max(0.15, annotated.shape[1] / 600)), (245, 245, 245), 1, cv2.LINE_AA,
    )
    return annotated
