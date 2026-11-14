"""Fixture replay is validated, visibly simulated, and entirely local."""

import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.ai.object_detection import DetectionError, FixtureDetector, filter_detections, load_labels, render_detections
from vidi.core.configuration import DetectionSettings
from vidi.core.types import Detection, DetectionResult, FramePacket


def fixture_document():
    return {
        "version": 1, "image_size": [80, 60],
        "detections": [{"class_id": 3, "label": "chair", "confidence": 0.8,
                        "bbox_xyxy": [10, 12, 40, 55]}],
    }


def write_fixture(tmp_path, document):
    path = tmp_path / "detections.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def make_packet(frame_id=7):
    return FramePacket(frame_id, 100, np.zeros((60, 80, 3), dtype=np.uint8))


def test_fixture_results_identify_simulation_and_current_frame(tmp_path):
    detector = FixtureDetector(write_fixture(tmp_path, fixture_document()), DetectionSettings())
    first, second = detector.detect(make_packet(7)), detector.detect(make_packet(8))
    assert first.backend == "fixture"
    assert first.status == "ready"
    assert first.simulated is True
    assert (first.frame_id, second.frame_id) == (7, 8)
    assert first.detections[0].class_id == 3
    assert first.detections[0].bbox_xyxy == (10, 12, 40, 55)


def test_fixture_filters_threshold_and_limits_in_descending_confidence(tmp_path):
    document = fixture_document()
    document["detections"] += [
        {"class_id": 1, "label": "person", "confidence": 0.95, "bbox_xyxy": [1, 2, 30, 40]},
        {"class_id": 2, "label": "door", "confidence": 0.2, "bbox_xyxy": [1, 2, 30, 40]},
    ]
    detector = FixtureDetector(
        write_fixture(tmp_path, document), DetectionSettings(confidence_threshold=0.5, max_detections=1)
    )
    assert [item.label for item in detector.detect(make_packet()).detections] == ["person"]


def test_sparse_labels_are_filtered_without_renumbering(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("-\nperson\n\nchair\n", encoding="utf-8")
    document = fixture_document()
    document["detections"] += [
        {"class_id": index, "label": "ignored fixture label", "confidence": 0.9,
         "bbox_xyxy": [1, 2, 30, 40]} for index in (0, 2, 99)
    ]
    detector = FixtureDetector(write_fixture(tmp_path, document), DetectionSettings(labels_path=labels))
    result = detector.detect(make_packet())
    assert [(item.class_id, item.label) for item in result.detections] == [(3, "chair")]
    assert load_labels(labels) == ("-", "person", "", "chair")


@pytest.mark.parametrize("label", ["", "-", "???", "unknown", "__background__"])
def test_placeholder_labels_never_become_detections(tmp_path, label):
    document = fixture_document()
    document["detections"][0]["label"] = label
    detector = FixtureDetector(write_fixture(tmp_path, document), DetectionSettings())
    result = detector.detect(make_packet())
    assert result.status == "ready"
    assert result.detections == ()


@pytest.mark.parametrize("size", [(61, 80, 3), (60, 81, 3), (60, 80)])
def test_fixture_rejects_differently_sized_images(tmp_path, size):
    detector = FixtureDetector(write_fixture(tmp_path, fixture_document()), DetectionSettings())
    with pytest.raises(DetectionError, match="Fixture expects"):
        detector.detect(FramePacket(1, 0, np.zeros(size, dtype=np.uint8)))


@pytest.mark.parametrize("field,value", [
    ("class_id", True), ("class_id", -1), ("label", 2), ("confidence", True),
    ("confidence", float("nan")), ("confidence", 1.1), ("confidence", 10**500),
    ("bbox_xyxy", [0, 0, 100, 60]), ("bbox_xyxy", [10, 10, 10, 20]),
    ("bbox_xyxy", [1.0, 2, 30, 40]),
])
def test_fixture_rejects_invalid_entries(tmp_path, field, value):
    document = fixture_document()
    document["detections"][0][field] = value
    with pytest.raises(DetectionError):
        FixtureDetector(write_fixture(tmp_path, document), DetectionSettings())


@pytest.mark.parametrize("document", [[], {}, {"version": 1},
    {"version": 2, "image_size": [80, 60], "detections": []},
    {"version": True, "image_size": [80, 60], "detections": []},
    {"version": 1, "image_size": [0, 60], "detections": []},
    {"version": 1, "image_size": [80, 60], "detections": {}},
])
def test_fixture_rejects_unsupported_schema(tmp_path, document):
    with pytest.raises(DetectionError):
        FixtureDetector(write_fixture(tmp_path, document), DetectionSettings())


def test_missing_or_corrupt_fixture_is_controlled_failure(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(DetectionError, match="Cannot read detection fixture"):
        FixtureDetector(path, DetectionSettings())
    path.write_text("{", encoding="utf-8")
    with pytest.raises(DetectionError, match="Cannot read detection fixture"):
        FixtureDetector(path, DetectionSettings())


def test_overlay_returns_copy_and_always_marks_fixture_simulated(monkeypatch, tmp_path):
    document = fixture_document()
    document["detections"] = []
    detector = FixtureDetector(write_fixture(tmp_path, document), DetectionSettings())
    packet = make_packet()
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", put_text)
    annotated = render_detections(packet.image_bgr, detector.detect(packet))
    assert annotated is not packet.image_bgr
    assert not np.any(packet.image_bgr)
    assert np.any(annotated)
    assert any(call.args[1] == "SIMULATED DETECTIONS" for call in put_text.call_args_list)


def test_local_filter_excludes_unscored_objects_without_assigning_confidence():
    cloud_object = Detection(None, "door", None, (1, 2, 30, 40))
    local_object = Detection(15, "person", 0.9, (10, 12, 40, 55))
    result = filter_detections((cloud_object, local_object), DetectionSettings())
    assert result == (local_object,)
    assert cloud_object.confidence is None and cloud_object.class_id is None


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -float("inf"), -0.1, 1.1, True, "0.9", 10**500])
def test_local_filter_excludes_invalid_scores(score):
    detection = Detection(15, "person", score, (1, 2, 30, 40))
    assert filter_detections((detection,), DetectionSettings(confidence_threshold=0)) == ()


def test_local_filter_preserves_threshold_order_limit_and_tie_stability():
    detections = (
        Detection(1, "chair", 0.55, (1, 2, 30, 40)),
        Detection(2, "person", 0.95, (1, 2, 30, 40)),
        Detection(3, "bicycle", 0.95, (1, 2, 30, 40)),
        Detection(4, "car", 0.54, (1, 2, 30, 40)),
        Detection(5, "unknown", 1.0, (1, 2, 30, 40)),
    )
    result = filter_detections(detections, DetectionSettings(max_detections=3))
    assert [item.label for item in result] == ["person", "bicycle", "chair"]


def test_cloud_overlay_uses_unscored_label_current_xyxy_and_owned_copy(monkeypatch):
    image = np.full((60, 80, 3), 15, dtype=np.uint8)
    original = image.copy()
    result = DetectionResult(
        7, "gemini", "ready", (Detection(None, "door", None, (10, 12, 40, 55)),),
    )
    put_text = Mock(wraps=cv2.putText)
    rectangle = Mock(wraps=cv2.rectangle)
    monkeypatch.setattr(cv2, "putText", put_text)
    monkeypatch.setattr(cv2, "rectangle", rectangle)

    annotated = render_detections(image, result)

    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(annotated, image)
    assert np.any(annotated != original)
    assert rectangle.call_args_list[0].args[1:3] == ((10, 12), (39, 54))
    assert [call.args[1] for call in put_text.call_args_list] == ["door", "GEMINI SNAPSHOT - PROTOTYPE"]
    assert result.detections[0].confidence is None and result.detections[0].class_id is None


def test_cloud_overlay_never_formats_a_score_as_calibrated_confidence(monkeypatch):
    result = DetectionResult(7, "gemini", "ready", (Detection(None, "chair", 0.9, (1, 2, 30, 40)),))
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", put_text)
    render_detections(make_packet().image_bgr, result)
    assert put_text.call_args_list[0].args[1] == "chair"


@pytest.mark.parametrize("status, outcome", [
    ("unavailable", "UNAVAILABLE"), ("failed", "FAILED"),
    ("cancelled", "CANCELLED"), ("not_ready", "NOT READY"),
])
def test_cloud_failure_overlay_is_explicit_and_never_shows_stale_boxes(monkeypatch, status, outcome):
    result = DetectionResult(7, "gemini", status, (Detection(None, "stale chair", None, (10, 12, 40, 55)),))
    put_text = Mock(wraps=cv2.putText)
    rectangle = Mock(wraps=cv2.rectangle)
    monkeypatch.setattr(cv2, "putText", put_text)
    monkeypatch.setattr(cv2, "rectangle", rectangle)

    render_detections(make_packet().image_bgr, result)

    assert [call.args[1] for call in put_text.call_args_list] == [f"GEMINI SNAPSHOT {outcome} - PROTOTYPE"]
    assert rectangle.call_count == 1  # Banner only, no stale bounding boxes.


def test_empty_ready_cloud_overlay_keeps_snapshot_identity(monkeypatch):
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", put_text)
    render_detections(make_packet().image_bgr, DetectionResult(7, "gemini", "ready"))
    assert put_text.call_args_list[0].args[1] == "GEMINI SNAPSHOT - PROTOTYPE"


@pytest.mark.parametrize("status, expected", [
    ("ready", "OBJECT DETECTION - PROTOTYPE"), ("not_ready", "INFERENCE NOT READY"),
])
def test_existing_local_banners_and_numeric_confidence_are_preserved(monkeypatch, status, expected):
    detections = (Detection(15, "person", 0.95, (10, 12, 40, 55)),) if status == "ready" else ()
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", put_text)
    render_detections(make_packet().image_bgr, DetectionResult(7, "opencv_ssd", status, detections))
    texts = [call.args[1] for call in put_text.call_args_list]
    assert texts[-1] == expected
    if detections:
        assert texts[0] == "person 0.95"
