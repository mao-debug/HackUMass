#!/usr/bin/env python3
"""Inspect assistant prerequisites without recording, playback or network access."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/pi-phase6.yaml"))
    args = parser.parse_args(argv)
    from vidi.core.configuration import load_settings
    from vidi.audio.wake_word import MODEL_FILES
    settings = load_settings(args.config)
    assets = settings.assistant
    modules = {}
    for name in ("sherpa_onnx", "chess", "picamera2", "cv2"):
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = False
    source = None
    if shutil.which("pactl"):
        try:
            result = subprocess.run(["pactl", "get-default-source"], capture_output=True, timeout=2., check=False)
            if result.returncode == 0:
                source = result.stdout[:1024].decode("utf-8").strip()
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            pass
    directory = assets.wake_model_dir_path
    keyboard = assets.keyboard_device_path
    try:
        import cv2
        aruco = hasattr(cv2, "aruco")
    except ImportError:
        aruco = False
    print(json.dumps({
        "python": sys.executable, "modules": modules, "aruco": aruco,
        "commands": {name: shutil.which(name) for name in ("arecord", "aplay", "paplay", "espeak-ng")},
        "default_input_is_non_monitor": bool(source and not source.endswith(".monitor")),
        "wake_files_present": bool(directory and all((directory / name).is_file() for name in MODEL_FILES.values())),
        "keywords_present": bool(assets.wake_keywords_path and assets.wake_keywords_path.is_file()),
        "piper_runtime_present": bool(settings.speech.piper_python_path and settings.speech.piper_python_path.is_file()),
        "piper_voice_present": bool(settings.speech.piper_model_path and settings.speech.piper_model_path.is_file()),
        "calibration_present": bool(settings.chess.calibration_path and settings.chess.calibration_path.is_file()),
        "selected_keyboard": str(keyboard) if keyboard else None,
        "keyboard_readable": bool(keyboard and keyboard.exists() and os.access(keyboard, os.R_OK)),
        "available_keyboard_links": sorted(map(str, Path("/dev/input/by-id").glob("*-event-kbd"))),
        "note": "Discovery only. No camera/microphone opened, speech played, service changed or network contacted."
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
