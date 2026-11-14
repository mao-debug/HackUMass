#!/usr/bin/env python3
"""Report local development prerequisites without opening hardware devices.

This diagnostic uses only the Python standard library. Finding a package,
executable, or device path does not verify that physical hardware works.
"""

from __future__ import annotations

import argparse
import importlib.metadata
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
    "libcamera",
    "pytesseract",
    "sounddevice",
    "vosk",
    "pyttsx3",
)
OPTIONAL_DISTRIBUTIONS = (
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "PyYAML",
    "picamera2",
    "google-genai",
    "setuptools",
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


def module_origin(name: str) -> str | None:
    """Locate a top-level module without loading its code or native library.

    Dotted module names are excluded because finding them can import their
    parent package. A namespace package may have no single origin.
    """
    if "." in name:
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None
    return None if spec is None else spec.origin


def distribution_version(name: str) -> str | None:
    """Read installed distribution metadata; missing metadata is inconclusive.

    In particular, an apt-provided cv2 module need not advertise the PyPI
    distribution name opencv-python. No module import or network lookup occurs.
    """
    try:
        return importlib.metadata.version(name)
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return None


def linux_distribution() -> dict[str, str] | None:
    """Read OS identity from the local os-release file when available."""
    try:
        values = platform.freedesktop_os_release()
    except (OSError, AttributeError):
        return None
    return {key: values[key] for key in ("ID", "VERSION_ID", "PRETTY_NAME") if key in values}


def collect_report(device_directory: Path = Path("/dev")) -> dict[str, Any]:
    """Collect hints about this computer; leave physical devices unverified."""
    try:
        camera_paths = sorted(str(path) for path in device_directory.glob("video*"))
    except OSError:
        camera_paths = []

    modules = {name: module_available(name) for name in OPTIONAL_MODULES}
    return {
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "virtual_environment": sys.prefix != sys.base_prefix,
            "linux_distribution": linux_distribution(),
        },
        "modules": modules,
        "module_origins": {
            name: module_origin(name) if available else None
            for name, available in modules.items()
        },
        "distribution_versions": {
            name: distribution_version(name) for name in OPTIONAL_DISTRIBUTIONS
        },
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
            "Distribution versions come from local metadata; missing metadata "
            "does not mean an apt-provided module is absent. Module origins do "
            "not verify NumPy/OpenCV ABI or camera/model compatibility. "
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
        f"Virtual environment: {system['virtual_environment']}",
        "Linux distribution: " + (
            json.dumps(system["linux_distribution"], sort_keys=True)
            if system["linux_distribution"] is not None else "unavailable"
        ),
        "",
        "Discoverable Python modules (future-phase modules are optional):",
    ]
    lines.extend(
        f"  {name}: {'found' if available else 'not found'}"
        + (f"; origin: {report['module_origins'][name]}" if report["module_origins"][name] else "")
        for name, available in report["modules"].items()
    )
    lines.append("Installed distribution versions (metadata only):")
    lines.extend(
        f"  {name}: {version or 'metadata unavailable'}"
        for name, version in report["distribution_versions"].items()
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
