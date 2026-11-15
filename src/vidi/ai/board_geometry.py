"""Local four-marker board calibration; absent geometry never reuses a crop.

The calibration fixes image coordinates, not piece identities. It requires an
OpenCV build with ArUco support and four DICT_4X4_50 markers, IDs 0 through 3.
No module in this file downloads assets or opens a camera.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MARKER_IDS = (0, 1, 2, 3)
MAX_CALIBRATION_BYTES = 32 * 1024


class BoardGeometryError(RuntimeError):
    """The current frame cannot establish trustworthy board coordinates."""


def _image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 32 or max(image.shape[:2]) > 16384):
        raise BoardGeometryError("Board geometry requires a bounded uint8 BGR image.")
    return image.shape[1], image.shape[0]


def _aruco():
    api = getattr(cv2, "aruco", None)
    if api is None:
        raise BoardGeometryError("This OpenCV build has no ArUco support. Keep automatic chess disabled.")
    dictionary = api.getPredefinedDictionary(api.DICT_4X4_50)
    return api, dictionary


def detect_markers(image) -> dict[int, np.ndarray]:
    """Find all four unique markers using new or OpenCV 4.6 legacy APIs."""
    width, height = _image(image)
    api, dictionary = _aruco()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(api, "ArucoDetector"):
        corners, ids, _ = api.ArucoDetector(dictionary, api.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = api.detectMarkers(gray, dictionary, parameters=api.DetectorParameters_create())
    found = {}
    if ids is not None:
        for marker, value in zip(corners, np.asarray(ids).reshape(-1)):
            key = int(value)
            if key not in MARKER_IDS:
                continue
            points = np.asarray(marker, dtype=np.float64).reshape(-1, 2)
            if key in found:
                raise BoardGeometryError("A required marker ID appears more than once.")
            _quad(points, width, height, minimum_area=64)
            found[key] = points
    if set(found) != set(MARKER_IDS):
        raise BoardGeometryError("All four unique board markers must be visible in the current frame.")
    return found


def _quad(points, width, height, *, minimum_area=256):
    if (not isinstance(points, (list, tuple, np.ndarray)) or len(points) != 4
            or any(not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2
                   or any(type(value) not in (int, float, np.int32, np.int64, np.float32, np.float64)
                          for value in point) for point in points)):
        raise BoardGeometryError("Board/marker corners must be numeric coordinate pairs.")
    points = np.asarray(points, dtype=np.float64)
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or (points < 0).any() or (points[:, 0] >= width).any()
            or (points[:, 1] >= height).any()):
        raise BoardGeometryError("Board/marker corners must be finite and inside the frame.")
    contour = points.astype(np.float32)
    area = cv2.contourArea(contour, oriented=True)
    if (area < minimum_area or not cv2.isContourConvex(contour)
            or np.min(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)) < 8):
        raise BoardGeometryError("Board/marker corners must form a clear clockwise quadrilateral.")
    return points


def create_calibration(image, board_quad, *, orientation="white_near", output_size=800) -> dict:
    """Capture a reference marker plane and clicked TL/TR/BR/BL board corners."""
    width, height = _image(image)
    if orientation not in ("white_near", "black_near"):
        raise BoardGeometryError("Choose a known white_near or black_near orientation.")
    if type(output_size) is not int or not 256 <= output_size <= 2048:
        raise BoardGeometryError("Rectified board size must be an integer from 256 to 2048.")
    quad = _quad(board_quad, width, height)
    markers = detect_markers(image)
    # The marker centers must surround the board, not form an arbitrary cluster.
    centers = np.array([markers[key].mean(axis=0) for key in MARKER_IDS])
    _quad(centers, width, height)
    if any(cv2.pointPolygonTest(centers.astype(np.float32), tuple(point), False) < 0 for point in quad):
        raise BoardGeometryError("Place marker centers outside all four board corners, ordered 0 TL, 1 TR, 2 BR, 3 BL.")
    return {"v": 1, "dictionary": "DICT_4X4_50", "orientation": orientation,
            "reference_size": [width, height], "output_size": output_size,
            "markers": {str(key): markers[key].tolist() for key in MARKER_IDS},
            "board_quad": quad.tolist()}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


class BoardRectifier:
    """Rectify only a new frame with all four consistent reference markers."""

    def __init__(self, calibration_path: Path | str):
        try:
            with Path(calibration_path).open("rb") as source:
                raw = source.read(MAX_CALIBRATION_BYTES + 1)
            if len(raw) > MAX_CALIBRATION_BYTES:
                raise ValueError
            document = json.loads(raw, object_pairs_hook=_unique,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if (not isinstance(document, dict) or set(document) != {
                    "v", "dictionary", "orientation", "reference_size", "output_size", "markers", "board_quad"}
                    or type(document["v"]) is not int or document["v"] != 1
                    or document["dictionary"] != "DICT_4X4_50"
                    or document["orientation"] not in ("white_near", "black_near")
                    or not isinstance(document["reference_size"], list) or len(document["reference_size"]) != 2
                    or any(type(x) is not int or not 32 <= x <= 16384 for x in document["reference_size"])
                    or type(document["output_size"]) is not int or not 256 <= document["output_size"] <= 2048
                    or not isinstance(document["markers"], dict) or set(document["markers"]) != set(map(str, MARKER_IDS))):
                raise ValueError
            width, height = document["reference_size"]
            self._reference = {key: _quad(document["markers"][str(key)], width, height, minimum_area=64) for key in MARKER_IDS}
            self._board = _quad(document["board_quad"], width, height)
            centers = np.array([self._reference[key].mean(axis=0) for key in MARKER_IDS])
            _quad(centers, width, height)
            if any(cv2.pointPolygonTest(centers.astype(np.float32), tuple(point), False) < 0 for point in self._board):
                raise ValueError
            self.orientation, self.output_size = document["orientation"], document["output_size"]
            self._reference_size = width, height
        except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError, cv2.error):
            raise BoardGeometryError("Board calibration is missing, invalid, or too large. Calibrate before enabling automatic chess.") from None

    def rectify(self, image) -> np.ndarray:
        """Return a new square BGR image or abstain; no last-good geometry cache."""
        width, height = _image(image)
        markers = detect_markers(image)
        current = np.concatenate([markers[key] for key in MARKER_IDS])
        reference = np.concatenate([self._reference[key] for key in MARKER_IDS])
        try:
            transform, _ = cv2.findHomography(current, reference, method=0)
            if transform is None or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-10:
                raise BoardGeometryError("Board marker geometry is degenerate.")
            projected = cv2.perspectiveTransform(current.reshape(1, -1, 2), transform)[0]
            if np.max(np.linalg.norm(projected - reference, axis=1)) > 3.0:
                raise BoardGeometryError("Board markers do not agree on one plane. Recalibrate the fixed board layout.")
            current_board = cv2.perspectiveTransform(self._board.reshape(1, 4, 2), np.linalg.inv(transform))[0]
            _quad(current_board, width, height)
            edge = self.output_size - 1
            destination = np.array([[0, 0], [edge, 0], [edge, edge], [0, edge]], np.float32)
            board_transform = cv2.getPerspectiveTransform(self._board.astype(np.float32), destination)
            return cv2.warpPerspective(image, board_transform @ transform,
                                       (self.output_size, self.output_size)).copy()
        except (cv2.error, np.linalg.LinAlgError, ValueError, OverflowError):
            raise BoardGeometryError("The current frame cannot establish reliable board coordinates.") from None


def marker_svg() -> str:
    """Generate a printable vector sheet from OpenCV's own local dictionary."""
    api, dictionary = _aruco()
    names = ("0: TOP LEFT", "1: TOP RIGHT", "2: BOTTOM RIGHT", "3: BOTTOM LEFT")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="210mm" viewBox="0 0 210 210">',
             '<rect width="210" height="210" fill="white"/>']
    for key in MARKER_IDS:
        tile = (api.generateImageMarker(dictionary, key, 60) if hasattr(api, "generateImageMarker")
                else api.drawMarker(dictionary, key, 60))
        x, y = 15 + (key % 2) * 100, 10 + (key // 2) * 100
        parts.append(f'<text x="{x}" y="{y + 80}" font-size="5">{names[key]}</text>')
        # Six cells: 4 data cells plus the required one-cell black border.
        for row in range(6):
            for col in range(6):
                if tile[row * 10 + 5, col * 10 + 5] == 0:
                    parts.append(f'<rect x="{x + col * 10}" y="{y + row * 10}" width="10" height="10" fill="black"/>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"
