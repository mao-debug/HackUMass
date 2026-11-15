"""Explicit upload/input combinations fail before any capture or SDK call."""

import pytest

from vidi.main import main


@pytest.mark.parametrize("arguments", [
    ["--mode", "chess"],
    ["--cloud-speech"],
    ["--assistant", "--cloud-speech"],
    ["--assistant", "--offline", "--cloud"],
    ["--assistant", "--describe"],
    ["--assistant", "--run-seconds", "nan"],
    ["--assistant", "--run-seconds", "0"],
    ["--assistant", "--automatic-chess"],
    ["--assistant", "--sensor-port", "/dev/ttyUSB0"],
    ["--assistant", "--detector", "gemini"],
    ["--assistant", "--max-frames", "1"],
])
def test_invalid_assistant_combinations(arguments):
    assert main(arguments) == 1


def test_assistant_does_not_require_continuous_or_preview(monkeypatch):
    import vidi.core.assistant_runtime as runtime
    seen = []
    monkeypatch.setattr(runtime, "run_assistant", lambda settings, **options: (
        seen.append((settings, options)) or {"frames_processed": 0, "capture_fps": None}))
    assert main(["--assistant", "--no-wake", "--offline", "--run-seconds", "1"]) == 0
    assert seen[0][1]["cloud_enabled"] is False
    assert seen[0][1]["wake"] is False
    assert seen[0][0].simulation.preview is False


def test_report_must_not_overwrite_saved_key_or_config(tmp_path, monkeypatch):
    from vidi.core import credentials
    key_path = tmp_path / "key"
    monkeypatch.setattr(credentials, "gemini_key_path", lambda: key_path)
    assert main(["--assistant", "--report", str(key_path)]) == 1
    config = tmp_path / "settings.yaml"
    config.write_text("models:\n  object_detection_backend: none\n")
    assert main(["--assistant", "--config", str(config), "--report", str(config)]) == 1
    assert config.read_text().startswith("models:")


def test_report_must_not_overwrite_wake_model_directory(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text("assistant:\n  wake_model_dir_path: models\n")
    assert main(["--assistant", "--config", str(config), "--report", str(tmp_path / "models" / "encoder.onnx")]) == 1
