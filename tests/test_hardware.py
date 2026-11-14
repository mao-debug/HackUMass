"""Hardware reporting must work without installing or opening peripherals."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_hardware.py"


@pytest.fixture
def hardware_module():
    spec = importlib.util.spec_from_file_location("visionbridge_hardware_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_keeps_physical_hardware_unverified(hardware_module, monkeypatch, tmp_path):
    (tmp_path / "video0").touch()
    (tmp_path / "video2").touch()
    (tmp_path / "unrelated").touch()
    monkeypatch.setattr(hardware_module, "module_available", lambda name: False)
    monkeypatch.setattr(hardware_module.shutil, "which", lambda name: None)

    report = hardware_module.collect_report(tmp_path)

    assert report["hardware"]["camera"] == {
        "status": "unverified",
        "device_paths": [str(tmp_path / "video0"), str(tmp_path / "video2")],
    }
    assert report["hardware"]["microphone"]["status"] == "unverified"
    assert report["hardware"]["audio_output"]["status"] == "unverified"
    assert not any(report["modules"].values())
    assert all(value is None for value in report["commands"].values())
    assert report["system"]["python"]


@pytest.mark.parametrize("error", [ImportError("missing"), ValueError("bad spec")])
def test_module_discovery_handles_broken_metadata(hardware_module, monkeypatch, error):
    def missing_spec(name):
        raise error

    monkeypatch.setattr(hardware_module.importlib.util, "find_spec", missing_spec)
    assert hardware_module.module_available("picamera2") is False


def test_module_discovery_uses_spec_only(hardware_module, monkeypatch):
    discovered = []

    def find_spec(name):
        discovered.append(name)
        return object()

    monkeypatch.setattr(hardware_module.importlib.util, "find_spec", find_spec)
    assert hardware_module.module_available("picamera2") is True
    assert discovered == ["picamera2"]


def test_json_cli_runs_with_standard_library_only():
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["hardware"]["camera"]["status"] == "unverified"
    assert report["hardware"]["microphone"]["status"] == "unverified"
    assert report["hardware"]["audio_output"]["status"] == "unverified"
    assert "No devices were opened" in report["note"]


def test_text_cli_is_readable(hardware_module, monkeypatch, capsys):
    monkeypatch.setattr(hardware_module, "module_available", lambda name: False)
    monkeypatch.setattr(hardware_module.shutil, "which", lambda name: None)
    assert hardware_module.main([]) == 0
    output = capsys.readouterr().out
    assert "VisionBridge local prerequisite report" in output
    assert "Physical camera: unverified" in output
    assert "Physical microphone: unverified" in output
    assert "Physical audio output: unverified" in output
