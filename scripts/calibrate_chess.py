#!/usr/bin/env python3
"""Generate local ArUco SVGs or calibrate a saved board image without uploads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

from vidi.ai.board_geometry import BoardGeometryError, create_calibration, marker_svg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--markers", type=Path, help="Write a printable four-marker SVG; no camera/network")
    mode.add_argument("--image", type=Path, help="Local board image with markers; click TL, TR, BR, BL board corners")
    parser.add_argument("--output", type=Path, help="New calibration JSON path for --image")
    parser.add_argument("--orientation", choices=("white_near", "black_near"), default="white_near")
    args = parser.parse_args(argv)
    if args.image is not None and args.output is None:
        parser.error("--image requires --output")
    if args.markers is not None and args.output is not None:
        parser.error("--markers is the output path; do not add --output")
    target = args.markers or args.output
    try:
        if target.exists():
            raise BoardGeometryError("Output already exists; choose a new path to preserve existing calibration/assets.")
        if args.markers is not None:
            content = marker_svg()
        else:
            if args.image.resolve() == target.resolve():
                raise BoardGeometryError("Calibration must not overwrite its input image.")
            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                raise BoardGeometryError("Cannot read the local calibration image.")
            # Detect before opening UI so a missing marker/build fails clearly.
            from vidi.ai.board_geometry import detect_markers
            detect_markers(image)
            corners = []
            window = "vidi calibration: click board TL TR BR BL; Enter saves; X clears; Q cancels"

            def click(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                    corners.append((x, y))

            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, click)
            try:
                while True:
                    display = image.copy()
                    for index, point in enumerate(corners):
                        cv2.circle(display, point, 5, (0, 255, 255), -1)
                        cv2.putText(display, str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                    cv2.imshow(window, display)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        return 130
                    if key in (ord("x"), ord("X")):
                        corners.clear()
                    if key in (10, 13) and len(corners) == 4:
                        content = json.dumps(create_calibration(image, corners, orientation=args.orientation), indent=2) + "\n"
                        break
            finally:
                cv2.destroyWindow(window)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as destination:
            destination.write(content)
        print("Saved local " + ("marker sheet: " if args.markers else "board calibration: ") + str(target))
        return 0
    except (BoardGeometryError, OSError, cv2.error):
        print("Calibration unavailable. Verify ArUco support, four unique markers, and ordered board corners; choose a new writable output path.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
