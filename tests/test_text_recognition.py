"""One-shot OCR uses fake subprocess results and actual local PNG encoding."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from vidi.ai import text_recognition as ocr
from vidi.ai.text_recognition import TesseractRecognizer
from vidi.core.configuration import OcrSettings
from vidi.core.local_process import LocalProcessError
from vidi.core.types import FramePacket


def packet(frame_id=7):
    image = np.full((40, 60, 3), (20, 80, 180), dtype=np.uint8)
    return FramePacket(frame_id, 123, image)


@pytest.fixture
def engine(monkeypatch):
    process = Mock(return_value=subprocess.CompletedProcess([], 0, b"MEETING ROOM 204\n", b""))
    monkeypatch.setattr(ocr, "run_local_process", process)
    return process


def test_constructor_does_not_start_engine(engine):
    TesseractRecognizer(OcrSettings())
    engine.assert_not_called()


def test_ocr_uses_fixed_arguments_stdin_image_bytes_and_grayscale(engine):
    frame = packet()
    original = frame.image_bgr.copy()

    result = TesseractRecognizer(OcrSettings()).recognize(frame)

    args, kwargs = engine.call_args
    assert args == (["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6", "--oem", "1"],)
    assert kwargs["timeout_s"] == 15.0
    assert kwargs["input_bytes"].startswith(b"\x89PNG\r\n\x1a\n")
    decoded = cv2.imdecode(np.frombuffer(kwargs["input_bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
    np.testing.assert_array_equal(frame.image_bgr, original)
    assert result.status == "ready" and result.text == "MEETING ROOM 204"
    assert result.backend == "tesseract" and result.frame_id == frame.frame_id
    assert result.error is None and result.elapsed_s >= 0


def test_no_preprocessing_preserves_original_bgr_png(engine):
    frame = packet()

    TesseractRecognizer(OcrSettings(preprocess="none")).recognize(frame)

    data = engine.call_args.kwargs["input_bytes"]
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, frame.image_bgr)


def test_custom_executable_is_one_literal_argument(engine):
    TesseractRecognizer(OcrSettings(executable="/local path/tesseract")).recognize(packet())
    assert engine.call_args.args[0][0] == "/local path/tesseract"


def test_requested_language_combination_is_one_argument(engine):
    TesseractRecognizer(OcrSettings(language="eng+spa")).recognize(packet())
    assert engine.call_args.args[0][4] == "eng+spa"


def test_preprocessing_owns_its_image(monkeypatch, engine):
    original_conversion = cv2.cvtColor
    frame = packet()
    original = frame.image_bgr.copy()

    def modify_copy(image, code):
        assert not np.shares_memory(image, frame.image_bgr)
        image[0, 0] = 0
        return original_conversion(image, code)

    monkeypatch.setattr(cv2, "cvtColor", modify_copy)
    assert TesseractRecognizer(OcrSettings()).recognize(frame).status == "ready"
    np.testing.assert_array_equal(frame.image_bgr, original)


def test_control_characters_are_cleaned_without_rewriting_words_or_paragraphs(engine):
    engine.return_value = subprocess.CompletedProcess(
        [], 0, "  First LINE.\r\n\r\nSecond\trow \x00\x07!\fCafé 42\r\n".encode(), b""
    )

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.text == "First LINE.\n\nSecond\trow !\n\nCafé 42"


def test_stderr_warnings_do_not_fail_successful_ocr(engine):
    engine.return_value = subprocess.CompletedProcess([], 0, b"EXIT 42\n", b"Estimating resolution as 224")

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.status == "ready" and result.text == "EXIT 42" and result.error is None


@pytest.mark.parametrize("text", [b"", b" \n\t\f\x00", b"\r\n\r\n"])
def test_empty_success_is_explicit_no_text(engine, text):
    engine.return_value = subprocess.CompletedProcess([], 0, text, b"warning")

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.status == "no_text"
    assert result.text == "" and result.error is None


def test_nonzero_exit_does_not_expose_raw_ocr_or_engine_diagnostics(engine, caplog):
    engine.return_value = subprocess.CompletedProcess([], 1, b"PRIVATE partial text", b"PRIVATE engine detail")

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.status == "failed" and result.text == ""
    assert "requested language data" in result.error
    assert "PRIVATE" not in result.error
    assert caplog.text == ""


@pytest.mark.parametrize("message", ["The requested local executable is unavailable.", "The local process timed out."])
def test_helper_failures_become_controlled_ocr_results(engine, message):
    engine.side_effect = LocalProcessError(message)

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.status == "failed"
    assert result.frame_id == 7 and result.backend == "tesseract"
    assert message in result.error and result.text == ""


def test_keyboard_interrupt_propagates_for_controller_cancellation(engine):
    engine.side_effect = KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        TesseractRecognizer(OcrSettings()).recognize(packet())


def test_non_utf8_engine_output_is_controlled_failure(engine):
    engine.return_value = subprocess.CompletedProcess([], 0, b"\xff\xfe", b"")

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.status == "failed" and result.text == ""
    assert "encoding" in result.error


@pytest.mark.parametrize(
    "image", [None, [], np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2), np.uint8),
              np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.float32)],
)
def test_invalid_frames_fail_without_starting_engine(engine, image):
    result = TesseractRecognizer(OcrSettings()).recognize(FramePacket(9, 0, image))
    assert result.status == "failed" and result.frame_id == 9
    engine.assert_not_called()


@pytest.mark.parametrize("stage", ["grayscale", "encode", "encode-false"])
def test_image_preparation_errors_are_controlled(monkeypatch, engine, stage):
    failure = cv2.error("PRIVATE OpenCV detail")
    if stage == "grayscale":
        monkeypatch.setattr(cv2, "cvtColor", Mock(side_effect=failure))
    elif stage == "encode":
        monkeypatch.setattr(cv2, "imencode", Mock(side_effect=failure))
    else:
        monkeypatch.setattr(cv2, "imencode", Mock(return_value=(False, None)))

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.status == "failed" and "PRIVATE" not in result.error
    engine.assert_not_called()


def test_each_call_returns_only_its_current_frame_text(engine):
    engine.side_effect = [
        subprocess.CompletedProcess([], 0, b"First frame", b""),
        subprocess.CompletedProcess([], 0, b"", b""),
    ]
    recognizer = TesseractRecognizer(OcrSettings())

    first, second = recognizer.recognize(packet(10)), recognizer.recognize(packet(11))

    assert first.frame_id == 10 and first.text == "First frame"
    assert second.frame_id == 11 and second.status == "no_text" and second.text == ""


def test_elapsed_time_covers_the_recognition_request(monkeypatch, engine):
    monkeypatch.setattr(ocr, "time", SimpleNamespace(monotonic=Mock(side_effect=[10.0, 10.75])))

    result = TesseractRecognizer(OcrSettings()).recognize(packet())

    assert result.elapsed_s == 0.75
