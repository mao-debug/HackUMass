#!/usr/bin/env python3
"""Report local development prerequisites without opening hardware devices.

This diagnostic uses only the Python standard library. Finding a package,
executable, or device path does not verify that physical hardware works.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any


OPTIONAL_MODULES = (
    "cv2",
    "numpy",
    "yaml",
    "picamera2",
    "pytesseract",
    "sounddevice",
    "vosk",
    "pyttsx3",
)
OPTIONAL_COMMANDS = (
    "rpicam-hello",
    "libcamera-hello",
    "tesseract",
    "espeak-ng",
    "say",
)


def module_available(name: str) -> bool:
    """Check discovery metadata without importing an optional dependency."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def collect_report(device_directory: Path = Path("/dev")) -> dict[str, Any]:
    """Collect hints about this computer; leave physical devices unverified."""
    try:
        camera_paths = sorted(str(path) for path in device_directory.glob("video*"))
    except OSError:
        camera_paths = []

    return {
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
        },
        "modules": {name: module_available(name) for name in OPTIONAL_MODULES},
        "commands": {name: shutil.which(name) for name in OPTIONAL_COMMANDS},
        "hardware": {
            "camera": {
                "status": "unverified",
                "device_paths": camera_paths,
            },
            "microphone": {"status": "unverified"},
            "audio_output": {"status": "unverified"},
        },
        "note": (
            "Package, executable, and device-path discovery does not verify "
            "physical hardware, permissions, driver support, or functionality. "
            "No devices were opened and no network connections were attempted."
        ),
    }


def format_report(report: dict[str, Any]) -> str:
    """Format a concise human-readable inventory of discovery results."""
    system = report["system"]
    lines = [
        "vidi local prerequisite report",
        f"OS: {system['os']} {system['release']}",
        f"Architecture: {system['architecture']}",
        f"Python: {system['python']} ({system['python_executable']})",
        "",
        "Discoverable Python modules (future-phase modules are optional):",
    ]
    lines.extend(
        f"  {name}: {'found' if available else 'not found'}"
        for name, available in report["modules"].items()
    )
    lines.append("Local executables:")
    lines.extend(
        f"  {name}: {path or 'not found'}"
        for name, path in report["commands"].items()
    )
    lines.extend(
        [
            "Physical camera: unverified",
            "Camera device paths: "
            + (", ".join(report["hardware"]["camera"]["device_paths"]) or "none found"),
            "Physical microphone: unverified",
            "Physical audio output: unverified",
            "",
            report["note"],
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print a report; absent optional software is informational, not failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print structured JSON")
    args = parser.parse_args(argv)
    report = collect_report()
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
