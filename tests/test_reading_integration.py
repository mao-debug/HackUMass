"""Optional real-engine checks; install engines only with separate approval."""

import json
import shutil
import wave

import cv2
import numpy as np
import pytest

from vidi.main import main
from vidi.audio.text_to_speech import LocalSpeech
from vidi.core.configuration import SpeechSettings


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Local Tesseract is not installed; package approval is separate.")
@pytest.mark.parametrize("printed", ["EXIT", "ROOM 201", "WELCOME TO VIDI"])
def test_real_tesseract_reads_generated_english_sign(tmp_path, printed):
    image = np.full((240, 1100, 3), 255, dtype=np.uint8)
    cv2.putText(image, printed, (45, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0, 0, 0), 5, cv2.LINE_AA)
    path, report = tmp_path / "printed.png", tmp_path / "reading.json"
    assert cv2.imwrite(str(path), image)
    assert main(["--image", str(path), "--read-text", "--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert " ".join(result["ocr"]["text"].split()) == printed
    assert result["ocr"]["status"] == "ready"
    assert result["speech"]["status"] == "not_requested"


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="Local eSpeak NG is not installed; package approval is separate.")
def test_real_espeak_exports_audio_without_playback(tmp_path):
    destination = tmp_path / "reading.wav"
    result = LocalSpeech(SpeechSettings(), "espeak_ng").speak("ROOM 201", output_path=destination)
    assert result.status == "completed", result.error
    assert result.mode == "wav" and result.output_path == destination
    with wave.open(str(destination), "rb") as audio:
        assert audio.getnframes() > 0
        assert audio.getcomptype() == "NONE"
