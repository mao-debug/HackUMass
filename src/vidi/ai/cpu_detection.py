"""Local OpenCV CPU inference for the original VOC MobileNet-SSD Caffe model.

Model/preprocessing reference: the model author's demonstration at
https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/demo.py
The supported contract is 300x300 BGR input, mean 127.5 and scale 0.007843,
with VOC's 21 class positions including background. Assets are local files;
this module never downloads a model or contacts a service.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vidi.ai.object_detection import DetectionError, filter_detections, load_labels
from vidi.core.types import Detection, DetectionResult, FramePacket

if TYPE_CHECKING:
    from vidi.core.configuration import DetectionSettings, ModelSettings


VOC_LABELS = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def _local_file(path: Path | None, description: str) -> Path:
    try:
        valid = path is not None and path.is_file()
    except OSError:
        valid = False
    if not valid:
        raise DetectionError(f"The OpenCV SSD detector requires an existing local {description} file.")
    return path


class OpenCvSsdDetector:
    """Run a configured Caffe MobileNet-SSD model on the local CPU only."""

    def __init__(self, models: ModelSettings, settings: DetectionSettings) -> None:
        model = _local_file(models.object_detection_path, "Caffe model")
        config = _local_file(models.object_detection_config_path, "Caffe prototxt")
        self.settings = settings
        self.labels = load_labels(settings.labels_path) if settings.labels_path else VOC_LABELS
        if len(self.labels) != len(VOC_LABELS):
            raise DetectionError("MobileNet-SSD labels must preserve all 21 VOC class positions, including background.")
        try:
            network = cv2.dnn.readNetFromCaffe(str(config), str(model))
            if network is None or network.empty():
                raise DetectionError("OpenCV could not load a nonempty MobileNet-SSD Caffe network.")
            network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as error:
            raise DetectionError(
                "Cannot load the local MobileNet-SSD Caffe model. Check that the prototxt and weights match."
            ) from error
        self._network = network

    def detect(self, packet: FramePacket) -> DetectionResult:
        """Return detections for this frame, including an explicit empty result."""
        image = packet.image_bgr
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or not image.size
        ):
            raise DetectionError("OpenCV SSD needs a nonempty, three-channel uint8 BGR frame.")
        try:
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=0.007843,
                size=(300, 300),
                mean=(127.5, 127.5, 127.5),
                swapRB=False,
                crop=False,
            )
            self._network.setInput(blob)
            output = self._network.forward()
        except cv2.error as error:
            raise DetectionError("OpenCV SSD inference failed. Check the local model and input configuration.") from error
        detections = self._decode(output, image.shape[1], image.shape[0])
        return DetectionResult(packet.frame_id, "opencv_ssd", "ready", detections, simulated=False)

    def _decode(self, output: object, width: int, height: int) -> tuple[Detection, ...]:
        if (
            not isinstance(output, np.ndarray)
            or output.dtype.kind not in ("f", "i", "u")
            or output.ndim != 4
            or output.shape[:2] != (1, 1)
            or output.shape[3] != 7
        ):
            raise DetectionError("OpenCV SSD requires numeric output tensors shaped [1,1,N,7].")
        if not np.isfinite(output).all():
            raise DetectionError("OpenCV SSD output contains non-finite values.")
        detections = []
        for row in output[0, 0]:
            image_id, category, score, x0, y0, x1, y1 = (float(value) for value in row)
            if image_id == -1:
                continue  # SSD's explicit no-detection sentinel.
            if image_id != 0:
                raise DetectionError("OpenCV SSD returned a detection for an unexpected image index.")
            if not 0 <= category < len(VOC_LABELS) or category != int(category):
                raise DetectionError("OpenCV SSD returned an invalid VOC class index.")
            if not 0 <= score <= 1:
                raise DetectionError("OpenCV SSD returned confidence outside [0,1].")
            class_id = int(category)
            if class_id == 0:
                continue  # A labels override must never turn background into an object.
            if x1 <= x0 or y1 <= y0:
                raise DetectionError("OpenCV SSD returned an empty or reversed bounding box.")
            # SSD can predict boxes crossing the image edge. Clip normalized
            # coordinates before multiplication to avoid overflow and retain
            # the output-image xyxy pixel convention used by other backends.
            x0, y0, x1, y1 = np.clip((x0, y0, x1, y1), 0.0, 1.0)
            left, top = int(x0 * width), int(y0 * height)
            right, bottom = int(x1 * width), int(y1 * height)
            if left >= right or top >= bottom:
                continue
            detections.append(Detection(
                class_id, self.labels[class_id].strip(), score, (left, top, right, bottom)
            ))
        return filter_detections(detections, self.settings)
