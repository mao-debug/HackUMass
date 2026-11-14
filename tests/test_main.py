"""Exercise the CLI without opening a camera, display, or network connection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys

import cv2
import pytest

from vidi import __version__
from vidi.camera.capture import CaptureError
from vidi.core import controller
from vidi.main import main


SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"


def run_module(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the real module entry point from outside the repository root."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_DIRECTORY)
    return subprocess.run(
        [sys.executable, "-m", "vidi", *arguments],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.fixture
def simulation_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "run_simulation", lambda settings: calls.append(settings))
    return calls


def test_module_default_sample_exports_one_frame_from_any_directory(tmp_path):
    result = run_module(tmp_path, "--output", "results/sample.png")

    assert result.returncode == 0, result.stderr
    assert "1 frame(s), 640x480, source=sample" in result.stderr
    assert "AI and audio are not active" in result.stderr
    frame = cv2.imread(str(tmp_path / "results/sample.png"))
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("--help", "vidi Phase 2"),
        ("--version", f"vidi {__version__}"),
    ],
)
def test_module_help_and_version_from_any_directory(tmp_path, argument, expected):
    result = run_module(tmp_path, argument)

    assert result.returncode == 0
    assert expected in result.stdout
    assert "Traceback" not in result.stderr


def test_default_cli_passes_bounded_hardware_free_settings(simulation_calls):
    assert main([]) == 0

    assert len(simulation_calls) == 1
    settings = simulation_calls[0]
    assert settings.camera.source == "sample"
    assert settings.simulation.max_frames == 1
    assert settings.simulation.preview is False


def test_yaml_paths_and_cli_overrides_use_their_respective_directories(
    tmp_path, monkeypatch, simulation_calls
):
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    config = config_directory / "settings.yaml"
    config.write_text(
        "camera:\n"
        "  source: image\n"
        "  image_path: samples/scene.png\n"
        "  width: 320\n"
        "  fps: 2\n"
        "simulation:\n"
        "  max_frames: 8\n"
        "  preview: true\n"
        "  output_path: results/config-frame.png\n"
        "log_level: WARNING\n",
        encoding="utf-8",
    )
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    status = main(
        ["--config", str(config), "--max-frames", "2", "--no-preview", "--log-level", "DEBUG"]
    )

    assert status == 0
    settings = simulation_calls[0]
    assert settings.camera.image_path == config_directory / "samples/scene.png"
    assert settings.camera.width == 320
    assert settings.camera.fps == 2
    assert settings.simulation.output_path == config_directory / "results/config-frame.png"
    assert settings.simulation.max_frames == 2
    assert settings.simulation.preview is False
    assert settings.log_level == "DEBUG"


def test_image_option_selects_image_and_resolves_cli_paths(
    tmp_path, monkeypatch, simulation_calls
):
    monkeypatch.chdir(tmp_path)

    assert main(["--image", "scene.png", "--output", "results/frame.png"]) == 0

    settings = simulation_calls[0]
    assert settings.camera.source == "image"
    assert settings.camera.image_path == tmp_path / "scene.png"
    assert settings.simulation.output_path == tmp_path / "results/frame.png"


def test_camera_index_selects_webcam_without_opening_hardware(simulation_calls):
    assert main(["--camera-index", "2"]) == 0

    assert simulation_calls[0].camera.source == "webcam"
    assert simulation_calls[0].camera.webcam_index == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-frames", "0"],
        ["--max-frames", "-1"],
        ["--camera-index", "-1"],
        ["--source", "image"],
        ["--image", "scene.png", "--camera-index", "0"],
        ["--source", "sample", "--image", "scene.png"],
        ["--source", "webcam", "--image", "scene.png"],
        ["--source", "sample", "--camera-index", "0"],
        ["--source", "image", "--camera-index", "0"],
    ],
)
def test_invalid_or_conflicting_options_fail_before_capture(
    arguments, simulation_calls, caplog
):
    assert main(arguments) == 1

    assert simulation_calls == []
    assert any(record.levelno == logging.ERROR for record in caplog.records)
    assert "Traceback" not in caplog.text


def test_missing_configuration_returns_controlled_failure(tmp_path, simulation_calls, caplog):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 1

    assert simulation_calls == []
    assert "Cannot read configuration" in caplog.text
    assert "Traceback" not in caplog.text


def test_missing_image_module_run_returns_failure_without_traceback(tmp_path):
    result = run_module(tmp_path, "--image", "missing.png")

    assert result.returncode == 1
    assert "missing.png" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("error_class", [CaptureError, controller.SimulationError])
def test_expected_capture_or_simulation_failures_are_reported(monkeypatch, caplog, error_class):
    def fail(settings):
        raise error_class("Controlled simulation failure")

    monkeypatch.setattr(controller, "run_simulation", fail)

    assert main([]) == 1
    assert "Controlled simulation failure" in caplog.text
    assert "Traceback" not in caplog.text


def test_missing_optional_runtime_dependency_is_reported(monkeypatch, caplog):
    def fail(settings):
        raise ImportError("unavailable local dependency")

    monkeypatch.setattr(controller, "run_simulation", fail)

    assert main([]) == 1
    assert "README.md" in caplog.text
    assert "Traceback" not in caplog.text


def test_keyboard_interrupt_returns_130_without_traceback(monkeypatch, caplog):
    def interrupt(settings):
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "run_simulation", interrupt)
    caplog.set_level(logging.INFO)

    assert main([]) == 130
    assert "Simulation stopped by the user" in caplog.text
    assert "Traceback" not in caplog.text
