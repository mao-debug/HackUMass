"""Synthetic marker planes exercise geometry without a camera or display."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vidi.ai import board_geometry as geometry


def markers():
    return {key: np.array([[x, y], [x + 40, y], [x + 40, y + 40], [x, y + 40]], float)
            for key, (x, y) in enumerate(((20, 20), (580, 20), (580, 580), (20, 580)))}


def image():
    yy, xx = np.indices((640, 640))
    return np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)


def calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(geometry, "detect_markers", lambda _: markers())
    data = geometry.create_calibration(image(), [[100, 100], [540, 100], [540, 540], [100, 540]], output_size=440)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data))
    return path


def test_rectification_owns_new_pixels_and_never_changes_input(tmp_path, monkeypatch):
    rectifier = geometry.BoardRectifier(calibration(tmp_path, monkeypatch))
    source = image()
    original = source.copy()
    rectified = rectifier.rectify(source)
    assert rectified.shape == (440, 440, 3) and rectified.dtype == np.uint8
    np.testing.assert_array_equal(rectified[0, 0], source[100, 100])
    assert not np.shares_memory(rectified, source)
    np.testing.assert_array_equal(source, original)
    assert rectifier.orientation == "white_near"


def test_missing_marker_never_reuses_last_good_geometry(tmp_path, monkeypatch):
    rectifier = geometry.BoardRectifier(calibration(tmp_path, monkeypatch))
    rectifier.rectify(image())
    def missing(_):
        raise geometry.BoardGeometryError("missing")
    monkeypatch.setattr(geometry, "detect_markers", missing)
    with pytest.raises(geometry.BoardGeometryError):
        rectifier.rectify(image())


def test_nonplanar_marker_movement_abstains(tmp_path, monkeypatch):
    rectifier = geometry.BoardRectifier(calibration(tmp_path, monkeypatch))
    displaced = markers()
    displaced[0] = displaced[0] + [60, 0]
    monkeypatch.setattr(geometry, "detect_markers", lambda _: displaced)
    with pytest.raises(geometry.BoardGeometryError, match="one plane"):
        rectifier.rectify(image())


def test_planar_translation_keeps_reference_board_coordinates(tmp_path, monkeypatch):
    rectifier = geometry.BoardRectifier(calibration(tmp_path, monkeypatch))
    shifted = {key: value + [10, 5] for key, value in markers().items()}
    monkeypatch.setattr(geometry, "detect_markers", lambda _: shifted)
    result = rectifier.rectify(image())
    np.testing.assert_array_equal(result[0, 0], image()[105, 110])


@pytest.mark.parametrize("legacy", [False, True])
def test_detection_supports_new_and_opencv46_legacy_api(monkeypatch, legacy):
    points = list(markers().values())
    calls = []
    def detect(*args, **kwargs):
        calls.append(1)
        return points, np.arange(4).reshape(4, 1), []
    api = SimpleNamespace(DICT_4X4_50=0, getPredefinedDictionary=lambda _: object())
    if legacy:
        api.DetectorParameters_create = lambda: object()
        api.detectMarkers = detect
    else:
        api.DetectorParameters = lambda: object()
        api.ArucoDetector = lambda *_: SimpleNamespace(detectMarkers=detect)
    monkeypatch.setattr(geometry.cv2, "aruco", api, raising=False)
    result = geometry.detect_markers(image())
    assert set(result) == {0, 1, 2, 3} and len(calls) == 1


@pytest.mark.parametrize("ids", [[0, 1, 2], [0, 1, 2, 2, 3]])
def test_missing_or_duplicate_required_ids_abstain(monkeypatch, ids):
    api = SimpleNamespace(DICT_4X4_50=0, getPredefinedDictionary=lambda _: None,
        DetectorParameters_create=lambda: None,
        detectMarkers=lambda *_args, **_kwargs: ([markers()[key] for key in ids], np.array(ids), []))
    monkeypatch.setattr(geometry.cv2, "aruco", api, raising=False)
    with pytest.raises(geometry.BoardGeometryError):
        geometry.detect_markers(image())


def test_missing_aruco_is_a_clear_optional_capability_failure(monkeypatch):
    monkeypatch.setattr(geometry.cv2, "aruco", None, raising=False)
    with pytest.raises(geometry.BoardGeometryError, match="no ArUco"):
        geometry.detect_markers(image())


@pytest.mark.parametrize("change", ["oversize", "duplicate", "nan", "unknown", "bad_quad", "bad_marker", "bad_size"])
def test_calibration_is_bounded_and_strict(tmp_path, monkeypatch, change):
    path = calibration(tmp_path, monkeypatch)
    document = json.loads(path.read_text())
    if change == "oversize":
        path.write_text("x" * (geometry.MAX_CALIBRATION_BYTES + 1))
    elif change == "duplicate":
        path.write_text('{"v":1,"v":1}')
    else:
        if change == "nan":
            document["board_quad"][0][0] = float("nan")
        elif change == "unknown":
            document["extra"] = 1
        elif change == "bad_quad":
            document["board_quad"] = [[10, 10]] * 4
        elif change == "bad_marker":
            document["markers"].pop("3")
        else:
            document["output_size"] = True
        path.write_text(json.dumps(document))
    with pytest.raises(geometry.BoardGeometryError):
        geometry.BoardRectifier(path)


def test_calibration_requires_marker_centers_outside_board_and_ordered_corners(monkeypatch):
    monkeypatch.setattr(geometry, "detect_markers", lambda _: markers())
    with pytest.raises(geometry.BoardGeometryError):
        geometry.create_calibration(image(), [[10, 10], [630, 10], [630, 630], [10, 630]])
    with pytest.raises(geometry.BoardGeometryError):
        geometry.create_calibration(image(), [[100, 100], [100, 540], [540, 540], [540, 100]])


def test_printable_svg_contains_four_distinct_local_markers(monkeypatch):
    calls = []
    def generate(dictionary, marker_id, size):
        calls.append(marker_id)
        tile = np.zeros((size, size), np.uint8)
        tile[10:20, (marker_id + 1) * 10:(marker_id + 2) * 10] = 255
        return tile
    monkeypatch.setattr(geometry, "_aruco", lambda: (SimpleNamespace(generateImageMarker=generate), object()))
    svg = geometry.marker_svg()
    assert svg.startswith('<svg') and calls == [0, 1, 2, 3]
    assert "TOP LEFT" in svg and "BOTTOM RIGHT" in svg and "http" not in svg.replace("http://www.w3.org/2000/svg", "")


def test_installed_aruco_synthetic_marker_roundtrip(tmp_path):
    api = getattr(geometry.cv2, "aruco", None)
    if api is None:
        pytest.skip("Optional ArUco capability is unavailable in this OpenCV build")
    dictionary = api.getPredefinedDictionary(api.DICT_4X4_50)
    source = np.full((640, 640, 3), 255, np.uint8)
    for key, (x, y) in enumerate(((10, 10), (570, 10), (570, 570), (10, 570))):
        tile = (api.generateImageMarker(dictionary, key, 60) if hasattr(api, "generateImageMarker")
                else api.drawMarker(dictionary, key, 60))
        source[y:y + 60, x:x + 60] = tile[:, :, None]
    found = geometry.detect_markers(source)
    assert set(found) == {0, 1, 2, 3}
    document = geometry.create_calibration(source, [[100, 100], [540, 100], [540, 540], [100, 540]])
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(document))
    rectifier = geometry.BoardRectifier(path)
    assert rectifier.rectify(source).shape == (800, 800, 3)
    source[565:635, 565:635] = 255
    with pytest.raises(geometry.BoardGeometryError):
        rectifier.rectify(source)
