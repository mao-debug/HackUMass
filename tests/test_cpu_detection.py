"""CPU detection contracts using a fake network and real local preprocessing."""

from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.ai.cpu_detection import OpenCvSsdDetector, VOC_LABELS
from vidi.ai.object_detection import DetectionError
from vidi.core.configuration import DetectionSettings, ModelSettings
from vidi.core.types import FramePacket


def output(*rows):
    return np.asarray(rows, dtype=np.float64).reshape(1, 1, len(rows), 7)


def packet(frame_id=7, image=None):
    if image is None:
        image = np.zeros((60, 80, 3), dtype=np.uint8)
    return FramePacket(frame_id, 123, image)


@pytest.fixture
def models(tmp_path):
    weights, config = tmp_path / "model.caffemodel", tmp_path / "model.prototxt"
    weights.write_bytes(b"Fake weights; the OpenCV model loader is mocked.")
    config.write_text("Fake configuration.", encoding="utf-8")
    return ModelSettings(
        object_detection_backend="opencv_ssd",
        object_detection_path=weights,
        object_detection_config_path=config,
    )


@pytest.fixture
def network(monkeypatch):
    net = Mock()
    net.empty.return_value = False
    net.forward.return_value = output()
    loader = Mock(return_value=net)
    monkeypatch.setattr(cv2.dnn, "readNetFromCaffe", loader)
    return net, loader


def test_model_is_loaded_from_local_files_and_forced_to_cpu(models, network):
    net, loader = network

    detector = OpenCvSsdDetector(models, DetectionSettings())
    detector.detect(packet())
    detector.detect(packet(8))

    loader.assert_called_once_with(
        str(models.object_detection_config_path), str(models.object_detection_path)
    )
    net.setPreferableBackend.assert_called_once_with(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget.assert_called_once_with(cv2.dnn.DNN_TARGET_CPU)
    assert net.forward.call_count == 2


@pytest.mark.parametrize("field", ["object_detection_path", "object_detection_config_path"])
@pytest.mark.parametrize("kind", ["missing", "directory", "unset"])
def test_missing_model_assets_fail_before_loading(models, network, tmp_path, field, kind):
    _, loader = network
    replacement = None if kind == "unset" else (tmp_path if kind == "directory" else tmp_path / "missing")

    with pytest.raises(DetectionError, match="existing local"):
        OpenCvSsdDetector(replace(models, **{field: replacement}), DetectionSettings())

    loader.assert_not_called()


@pytest.mark.parametrize("failure", ["load", "empty", "none", "backend", "target"])
def test_network_loading_or_cpu_configuration_failures_are_controlled(models, network, failure):
    net, loader = network
    if failure == "load":
        loader.side_effect = cv2.error("invalid Caffe model")
    elif failure == "empty":
        net.empty.return_value = True
    elif failure == "none":
        loader.return_value = None
    elif failure == "backend":
        net.setPreferableBackend.side_effect = cv2.error("backend unavailable")
    else:
        net.setPreferableTarget.side_effect = cv2.error("target unavailable")

    with pytest.raises(DetectionError):
        OpenCvSsdDetector(models, DetectionSettings())


def test_preprocessing_keeps_bgr_channels_and_does_not_mutate_frame(models, network):
    net, _ = network
    image = np.full((30, 50, 3), (0, 127, 255), dtype=np.uint8)
    original = image.copy()

    OpenCvSsdDetector(models, DetectionSettings()).detect(packet(image=image))

    blob = net.setInput.call_args.args[0]
    assert blob.shape == (1, 3, 300, 300)
    assert blob.dtype == np.float32
    expected = (np.array([0, 127, 255], dtype=np.float32) - 127.5) * 0.007843
    np.testing.assert_allclose(blob[0, :, 100, 100], expected, rtol=1e-6)
    assert not np.shares_memory(blob, image)
    np.testing.assert_array_equal(image, original)


def test_voc_class_indices_and_boxes_use_current_frame_dimensions(models, network):
    net, _ = network
    net.forward.return_value = output(
        [0, 15, 0.9, 0.25, 0.25, 0.75, 0.75],
        [0, 20, 0.8, 0, 0, 1, 1],
    )

    result = OpenCvSsdDetector(models, DetectionSettings()).detect(packet(42))

    assert result.frame_id == 42
    assert result.backend == "opencv_ssd"
    assert result.status == "ready"
    assert result.simulated is False
    assert [(d.class_id, d.label, d.bbox_xyxy) for d in result.detections] == [
        (15, "person", (20, 15, 60, 45)),
        (20, "tvmonitor", (0, 0, 80, 60)),
    ]
    assert result.detections[0].confidence == pytest.approx(0.9)


def test_scores_are_filtered_and_limited_in_confidence_order(models, network):
    net, _ = network
    net.forward.return_value = output(
        [0, 15, 0.6, 0, 0, 1, 1],
        [0, 9, 0.95, 0, 0, 1, 1],
        [0, 12, 0.2, 0, 0, 1, 1],
    )

    result = OpenCvSsdDetector(
        models, DetectionSettings(confidence_threshold=0.5, max_detections=1)
    ).detect(packet())

    assert [(d.class_id, d.label) for d in result.detections] == [(9, "chair")]


def test_threshold_is_inclusive(models, network):
    net, _ = network
    net.forward.return_value = output(
        [0, 15, 0.55, 0, 0, 1, 1], [0, 9, 0.54, 0, 0, 1, 1]
    )

    result = OpenCvSsdDetector(models, DetectionSettings()).detect(packet())

    assert [d.label for d in result.detections] == ["person"]


def test_background_and_ssd_sentinel_are_successful_empty_results(models, network):
    net, _ = network
    net.forward.return_value = output(
        [0, 0, 0, 0, 0, 0, 0], [-1, -1, -1, -1, -1, -1, -1]
    )

    result = OpenCvSsdDetector(models, DetectionSettings()).detect(packet())

    assert result.status == "ready"
    assert result.detections == ()
    assert result.simulated is False


def test_boxes_are_clipped_before_multiplication_and_outside_boxes_are_dropped(models, network):
    net, _ = network
    net.forward.return_value = output(
        [0, 15, 0.9, -1e300, -0.25, 1e300, 1.5],
        [0, 9, 0.8, -3, -3, -1, -1],
        [0, 12, 0.7, 2, 2, 3, 3],
    )

    result = OpenCvSsdDetector(models, DetectionSettings()).detect(packet())

    assert len(result.detections) == 1
    assert result.detections[0].bbox_xyxy == (0, 0, 80, 60)


@pytest.mark.parametrize("count", [20, 22])
def test_custom_labels_require_all_21_class_positions(models, network, tmp_path, count):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(["label"] * count) + "\n", encoding="utf-8")
    _, loader = network

    with pytest.raises(DetectionError, match="21 VOC class positions"):
        OpenCvSsdDetector(models, DetectionSettings(labels_path=labels))

    loader.assert_not_called()


def test_custom_labels_preserve_indices_and_cannot_relabel_background(models, network, tmp_path):
    labels = list(VOC_LABELS)
    labels[0], labels[9], labels[15], labels[20] = "not background", "-", "visitor", "display"
    path = tmp_path / "labels.txt"
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    net, _ = network
    net.forward.return_value = output(
        [0, 0, 1, 0, 0, 1, 1], [0, 9, 0.95, 0, 0, 1, 1],
        [0, 15, 0.9, 0, 0, 1, 1], [0, 20, 0.8, 0, 0, 1, 1],
    )

    result = OpenCvSsdDetector(models, DetectionSettings(labels_path=path)).detect(packet())

    assert [(d.class_id, d.label) for d in result.detections] == [(15, "visitor"), (20, "display")]


def test_missing_labels_are_controlled_before_model_loading(models, network, tmp_path):
    _, loader = network

    with pytest.raises(DetectionError, match="Cannot read labels"):
        OpenCvSsdDetector(models, DetectionSettings(labels_path=tmp_path / "missing.txt"))

    loader.assert_not_called()


@pytest.mark.parametrize(
    "invalid_output",
    [None, [], np.zeros((1, 7)), np.zeros((2, 1, 3, 7)), np.zeros((1, 2, 3, 7)),
     np.zeros((1, 1, 3, 6)), np.zeros((1, 1, 3, 7), dtype=object),
     np.zeros((1, 1, 3, 7), dtype=bool), np.zeros((1, 1, 3, 7), dtype=complex)],
)
def test_wrong_output_shape_or_type_is_a_controlled_error(models, network, invalid_output):
    net, _ = network
    net.forward.return_value = invalid_output

    with pytest.raises(DetectionError, match="shaped"):
        OpenCvSsdDetector(models, DetectionSettings()).detect(packet())


@pytest.mark.parametrize("column", range(7))
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_outputs_are_rejected_in_every_column(models, network, column, value):
    row = [0, 15, 0.9, 0, 0, 1, 1]
    row[column] = value
    net, _ = network
    net.forward.return_value = output(row)

    with pytest.raises(DetectionError, match="non-finite"):
        OpenCvSsdDetector(models, DetectionSettings()).detect(packet())


@pytest.mark.parametrize(
    ("column", "value"),
    [(0, 1), (0, -2), (0, 0.5), (1, -1), (1, 21), (1, 15.5), (2, -0.1), (2, 1.1)],
)
def test_invalid_single_image_class_and_score_ranges_are_rejected(models, network, column, value):
    row = [0, 15, 0.9, 0, 0, 1, 1]
    row[column] = value
    net, _ = network
    net.forward.return_value = output(row)

    with pytest.raises(DetectionError):
        OpenCvSsdDetector(models, DetectionSettings()).detect(packet())


@pytest.mark.parametrize("box", [[0.75, 0.25, 0.25, 0.75], [0.25, 0.75, 0.75, 0.25],
                                  [0.25, 0.25, 0.25, 0.75], [0.25, 0.25, 0.75, 0.25]])
def test_empty_or_reversed_boxes_are_rejected(models, network, box):
    net, _ = network
    net.forward.return_value = output([0, 15, 0.9, *box])

    with pytest.raises(DetectionError, match="empty or reversed"):
        OpenCvSsdDetector(models, DetectionSettings()).detect(packet())


@pytest.mark.parametrize(
    "image",
    [None, [], np.zeros((0, 20, 3), np.uint8), np.zeros((20, 20), np.uint8),
     np.zeros((20, 20, 4), np.uint8), np.zeros((20, 20, 3), np.float32)],
)
def test_invalid_frames_do_not_run_the_network(models, network, image):
    net, _ = network
    detector = OpenCvSsdDetector(models, DetectionSettings())

    with pytest.raises(DetectionError, match="uint8 BGR frame"):
        detector.detect(FramePacket(1, 0, image))

    net.setInput.assert_not_called()
    net.forward.assert_not_called()


@pytest.mark.parametrize("stage", ["preprocess", "set-input", "forward"])
def test_opencv_inference_failures_are_controlled(monkeypatch, models, network, stage):
    net, _ = network
    detector = OpenCvSsdDetector(models, DetectionSettings())
    failure = cv2.error("local OpenCV inference failure")
    if stage == "preprocess":
        monkeypatch.setattr(cv2.dnn, "blobFromImage", Mock(side_effect=failure))
    elif stage == "set-input":
        net.setInput.side_effect = failure
    else:
        net.forward.side_effect = failure

    with pytest.raises(DetectionError, match="inference failed"):
        detector.detect(packet())


def test_empty_and_failed_inference_never_reuses_previous_detections(models, network):
    net, _ = network
    net.forward.side_effect = [
        output([0, 15, 0.9, 0, 0, 1, 1]), output(), cv2.error("forward failed"), output(),
    ]
    detector = OpenCvSsdDetector(models, DetectionSettings())

    first = detector.detect(packet(101))
    empty = detector.detect(packet(102))
    with pytest.raises(DetectionError):
        detector.detect(packet(103))
    later = detector.detect(packet(104))

    assert first.frame_id == 101 and len(first.detections) == 1
    assert empty.frame_id == 102 and empty.detections == ()
    assert later.frame_id == 104 and later.detections == ()
    assert first.detections[0].label == "person"
    assert net.setInput.call_count == 4
