"""One-shot local Tesseract OCR using image bytes and fixed command arguments."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import unicodedata

import cv2
import numpy as np

from vidi.core.local_process import LocalProcessError, run_local_process
from vidi.core.types import FramePacket, OcrResult

if TYPE_CHECKING:
    from vidi.core.configuration import OcrSettings


def _clean_text(text: str) -> str:
    """Remove controls while retaining recognized words and paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    return "".join(
        character for character in text
        if character in ("\n", "\t") or not unicodedata.category(character).startswith("C")
    ).strip()


class TesseractRecognizer:
    """Recognize a single raw frame; construction starts no processes."""

    def __init__(self, settings: OcrSettings) -> None:
        self.settings = settings

    def recognize(self, packet: FramePacket) -> OcrResult:
        started = time.monotonic()

        def result(status: str, *, text: str = "", error: str | None = None) -> OcrResult:
            return OcrResult(
                status=status, frame_id=packet.frame_id, text=text, backend="tesseract",
                elapsed_s=max(0.0, time.monotonic() - started), error=error,
            )

        image = packet.image_bgr
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or not image.size
        ):
            return result("failed", error="OCR needs a nonempty, three-channel uint8 BGR camera image.")
        try:
            owned_image = np.array(image, copy=True, order="C")
            if self.settings.preprocess == "grayscale":
                owned_image = cv2.cvtColor(owned_image, cv2.COLOR_BGR2GRAY)
            elif self.settings.preprocess != "none":
                return result("failed", error="The requested OCR preprocessing mode is unsupported.")
            success, encoded = cv2.imencode(".png", owned_image)
            if not success:
                return result("failed", error="The camera image could not be encoded for local OCR.")
        except cv2.error:
            return result("failed", error="The camera image could not be prepared for local OCR.")

        try:
            completed = run_local_process(
                [
                    self.settings.executable, "stdin", "stdout", "-l", self.settings.language,
                    "--psm", str(self.settings.psm), "--oem", "1",
                ],
                input_bytes=encoded.tobytes(),
                timeout_s=self.settings.timeout_s,
            )
        except LocalProcessError as error:
            return result("failed", error=f"Local OCR could not complete: {error}")
        if completed.returncode != 0:
            return result(
                "failed",
                error="Tesseract OCR failed. Check the requested language data and image configuration.",
            )
        try:
            text = _clean_text(completed.stdout.decode("utf-8"))
        except UnicodeError:
            return result("failed", error="Tesseract returned text with an unsupported encoding.")
        return result("ready", text=text) if text else result("no_text")
