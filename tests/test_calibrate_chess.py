"""Calibration CLI writes local assets only; fake UI never opens a display."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("calibrate_chess", Path(__file__).resolve().parents[1] / "scripts/calibrate_chess.py")
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


def test_marker_mode_writes_only_requested_new_local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "marker_svg", lambda: "<svg/>\n")
    monkeypatch.setattr(calibrate.cv2, "namedWindow", lambda *_: pytest.fail("UI opened in marker mode"))
    output = tmp_path / "local" / "markers.svg"
    assert calibrate.main(["--markers", str(output)]) == 0
    assert output.read_text() == "<svg/>\n"
    output.write_text("keep existing")
    assert calibrate.main(["--markers", str(output)]) == 1
    assert output.read_text() == "keep existing"


def test_missing_image_fails_without_ui_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate.cv2, "imread", lambda *_: None)
    monkeypatch.setattr(calibrate.cv2, "namedWindow", lambda *_: pytest.fail("UI opened without image"))
    output = tmp_path / "calibration.json"
    assert calibrate.main(["--image", str(tmp_path / "missing.png"), "--output", str(output)]) == 1
    assert not output.exists()


def test_four_clicks_are_required_and_validated_before_local_save(tmp_path, monkeypatch):
    import vidi.ai.board_geometry as geometry
    monkeypatch.setattr(geometry, "detect_markers", lambda _: {})
    monkeypatch.setattr(calibrate.cv2, "imread", lambda *_: np.zeros((100, 100, 3), np.uint8))
    for name in ("namedWindow", "imshow", "circle", "putText", "destroyWindow"):
        monkeypatch.setattr(calibrate.cv2, name, Mock())
    def install_callback(window, callback):
        for x, y in ((10, 10), (90, 10), (90, 90), (10, 90)):
            callback(calibrate.cv2.EVENT_LBUTTONDOWN, x, y, None, None)
    monkeypatch.setattr(calibrate.cv2, "setMouseCallback", install_callback)
    monkeypatch.setattr(calibrate.cv2, "waitKey", lambda *_: 13)
    create = Mock(return_value={"v": 1})
    monkeypatch.setattr(calibrate, "create_calibration", create)
    output = tmp_path / "calibration.json"
    assert calibrate.main(["--image", "saved-board.png", "--output", str(output), "--orientation", "black_near"]) == 0
    assert create.call_args.args[1] == [(10, 10), (90, 10), (90, 90), (10, 90)]
    assert create.call_args.kwargs == {"orientation": "black_near"}
    assert output.read_text().endswith("\n")
    calibrate.cv2.destroyWindow.assert_called_once()
