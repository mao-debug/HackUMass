#!/usr/bin/env python3
"""Speak one fixed local voice sample without opening a camera or using an API."""

import argparse
import logging
from pathlib import Path

from vidi.audio.text_to_speech import LocalSpeech
from vidi.core.configuration import ConfigurationError, load_settings


SAMPLE_TEXT = (
    "This is a voice test for vidi. Please choose the voice that is easiest for you to hear. "
    "I can read printed signs, labels, and short paragraphs aloud."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Use only the speech settings and local speech backend from this YAML")
    parser.add_argument("--output", type=Path, help="Export a WAV instead of playing sound")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = load_settings(args.config)
        backend = settings.models.text_to_speech_backend
        if backend not in (None, "espeak_ng", "say", "piper"):
            raise ConfigurationError("Voice preview supports local speech engines only.")
        if args.output is not None and args.output.suffix.lower() != ".wav":
            raise ConfigurationError("Voice preview output must use a .wav extension.")
        result = LocalSpeech(settings.speech, backend).speak(
            SAMPLE_TEXT, output_path=args.output.expanduser().resolve() if args.output is not None else None,
        )
        if result.status != "completed":
            logging.error("Voice preview failed: %s", result.error or "Speech unavailable.")
            return 1
        logging.info("Voice sample completed with %s. %s", result.backend,
                     "WAV export only." if args.output is not None else "Confirm audibility through your selected audio output.")
        return 0
    except (ConfigurationError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.info("Voice preview stopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
