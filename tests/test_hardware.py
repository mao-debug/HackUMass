"""Hardware reporting must work without installing or opening peripherals."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_hardware.py"


@pytest.fixture
def hardware_module():
    spec = importlib.util.spec_from_file_location("vidi_hardware_check", SCRIPT)
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
    assert report["system"]["process_bits"] in (32, 64)
    assert report["system"]["board_model"] is None or isinstance(report["system"]["board_model"], str)
    assert report["system"]["memory_total_mb"] is None or isinstance(report["system"]["memory_total_mb"], int)


def test_text_cli_is_readable(hardware_module, monkeypatch, capsys):
    monkeypatch.setattr(hardware_module, "module_available", lambda name: False)
    monkeypatch.setattr(hardware_module.shutil, "which", lambda name: None)
    assert hardware_module.main([]) == 0
    output = capsys.readouterr().out
    assert "vidi local prerequisite report" in output
    assert "Physical camera: unverified" in output
    assert "Physical microphone: unverified" in output
    assert "Physical audio output: unverified" in output


def test_module_origin_reads_spec_without_importing_package(hardware_module, monkeypatch):
    inspected = []

    def find_spec(name):
        inspected.append(name)
        return SimpleNamespace(origin="/usr/lib/python3/dist-packages/picamera2/__init__.py")

    monkeypatch.setattr(hardware_module.importlib.util, "find_spec", find_spec)
    assert hardware_module.module_origin("picamera2") == (
        "/usr/lib/python3/dist-packages/picamera2/__init__.py"
    )
    assert inspected == ["picamera2"]
    assert hardware_module.module_origin("google.genai") is None
    assert inspected == ["picamera2"]


@pytest.mark.parametrize("spec", [None, SimpleNamespace(origin=None)])
def test_module_origin_can_be_absent(hardware_module, monkeypatch, spec):
    monkeypatch.setattr(hardware_module.importlib.util, "find_spec", lambda name: spec)
    assert hardware_module.module_origin("picamera2") is None


def test_distribution_version_uses_local_metadata(hardware_module, monkeypatch):
    monkeypatch.setattr(hardware_module.importlib.metadata, "version", lambda name: "1.24.2")
    assert hardware_module.distribution_version("numpy") == "1.24.2"


def test_missing_distribution_metadata_does_not_hide_module(hardware_module, monkeypatch, tmp_path):
    def no_metadata(name):
        raise hardware_module.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(hardware_module.importlib.metadata, "version", no_metadata)
    monkeypatch.setattr(hardware_module, "module_available", lambda name: name == "cv2")
    monkeypatch.setattr(hardware_module, "module_origin", lambda name: "/usr/lib/python3/dist-packages/cv2.so")
    report = hardware_module.collect_report(tmp_path)

    assert report["modules"]["cv2"] is True
    assert report["module_origins"]["cv2"] == "/usr/lib/python3/dist-packages/cv2.so"
    assert report["distribution_versions"]["opencv-python"] is None
    assert "missing metadata does not mean an apt-provided module is absent" in report["note"]
    assert report["hardware"]["camera"]["status"] == "unverified"


def test_linux_distribution_reads_selected_local_fields(hardware_module, monkeypatch):
    monkeypatch.setattr(
        hardware_module.platform,
        "freedesktop_os_release",
        lambda: {"ID": "debian", "VERSION_ID": "12", "PRETTY_NAME": "Debian", "EXTRA": "ignored"},
    )
    assert hardware_module.linux_distribution() == {
        "ID": "debian", "VERSION_ID": "12", "PRETTY_NAME": "Debian"
    }


def test_linux_distribution_can_be_unavailable(hardware_module, monkeypatch):
    def no_os_release():
        raise OSError("not a Linux host")

    monkeypatch.setattr(hardware_module.platform, "freedesktop_os_release", no_os_release)
    assert hardware_module.linux_distribution() is None


@pytest.mark.parametrize("model", ["Raspberry Pi 4 Model B Rev 1.5", "Raspberry Pi 5 Model B Rev 1.0"])
def test_board_model_reads_injected_nul_terminated_device_tree_metadata(hardware_module, tmp_path, model):
    path = tmp_path / "model"
    path.write_bytes(model.encode("utf-8") + b"\x00")
    assert hardware_module.board_model((path,)) == model


@pytest.mark.parametrize("primary_data", [None, b"\xff\xfe", b"Raspberry\x00Pi", b"x" * 257])
def test_missing_malformed_or_oversized_primary_model_uses_fallback(hardware_module, tmp_path, primary_data):
    primary, fallback = tmp_path / "primary-model", tmp_path / "fallback-model"
    if primary_data is not None:
        primary.write_bytes(primary_data)
    fallback.write_bytes(b"Raspberry Pi 5 Model B Rev 1.0\x00\x00")
    assert hardware_module.board_model((primary, fallback)) == "Raspberry Pi 5 Model B Rev 1.0"


def test_missing_or_empty_metadata_is_unavailable(hardware_module, tmp_path):
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.write_bytes(b"\x00")
    assert hardware_module.board_model((missing, empty)) is None
    assert hardware_module.memory_total_mb(missing) is None


def test_memory_total_is_kernel_reported_mib_not_a_board_capacity_assumption(hardware_module, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:        8090368 kB\nMemFree:          500000 kB\n")
    assert hardware_module.memory_total_mb(meminfo) == 7900


@pytest.mark.parametrize("data", [b"MemTotal: -1 kB", b"MemTotal: 8 MB", b"MemTotal: nope kB", b"\xff", b"MemFree: 1024 kB"])
def test_malformed_memory_metadata_is_unavailable(hardware_module, tmp_path, data):
    meminfo = tmp_path / "meminfo"
    meminfo.write_bytes(data)
    assert hardware_module.memory_total_mb(meminfo) is None


def test_report_adds_board_bitness_memory_and_discovery_without_running_commands(hardware_module, monkeypatch, tmp_path):
    model, meminfo = tmp_path / "model", tmp_path / "meminfo"
    model.write_bytes(b"Raspberry Pi 5 Model B Rev 1.0\x00")
    meminfo.write_text("MemTotal: 8388608 kB\n")
    monkeypatch.setattr(hardware_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(hardware_module.struct, "calcsize", lambda format: 4)
    monkeypatch.setattr(hardware_module, "module_available", lambda name: name == "piper")
    monkeypatch.setattr(hardware_module, "module_origin", lambda name: "/local/piper/__init__.py")
    monkeypatch.setattr(hardware_module, "distribution_version", lambda name: "test-version" if name == "piper-tts" else None)
    discovered = []

    def which(name):
        discovered.append(name)
        return f"/local/bin/{name}"

    def no_process(*args, **kwargs):
        pytest.fail("The diagnostic must not execute a hardware/audio command")

    monkeypatch.setattr(hardware_module.shutil, "which", which)
    monkeypatch.setattr(subprocess, "Popen", no_process)

    report = hardware_module.collect_report(tmp_path, model_paths=(model,), meminfo_path=meminfo)

    assert report["system"]["board_model"] == "Raspberry Pi 5 Model B Rev 1.0"
    assert report["system"]["architecture"] == "aarch64" and report["system"]["process_bits"] == 32
    assert report["system"]["memory_total_mb"] == 8192
    assert report["modules"]["piper"] is True and report["module_origins"]["piper"] == "/local/piper/__init__.py"
    assert report["distribution_versions"]["piper-tts"] == "test-version"
    assert {"vcgencmd", "bluetoothctl", "wpctl", "pactl", "paplay", "pw-play", "aplay", "afplay"} <= set(discovered)
    assert all(report["hardware"][name]["status"] == "unverified" for name in ("camera", "microphone", "audio_output"))
    text = hardware_module.format_report(report)
    assert "Board model: Raspberry Pi 5 Model B Rev 1.0" in text
    assert "Process bitness: 32-bit" in text and "Reported memory total: 8192 MiB" in text
    assert "do not verify Pi compatibility" in report["note"]
