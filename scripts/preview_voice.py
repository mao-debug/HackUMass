#!/usr/bin/env python3
"""Audition fixed speech samples; cloud TTS requires explicit enablement."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import logging
from pathlib import Path

from vidi.audio.text_to_speech import LocalSpeech
from vidi.core.configuration import ConfigurationError, load_settings
from vidi.core.credentials import CredentialError


SAMPLE_TEXT = (
    "This is a voice test for vidi. Please choose the voice that is easiest for you to hear. "
    "I can read printed signs, labels, and short paragraphs aloud."
)
CHESS_SAMPLE_TEXT = (
    "This is a chess voice test. White knight from g1 to f3. "
    "Black king castles from e8 to g8; rook from h8 to f8. "
    "e4 is empty. This is only a proposed move. Verify the physical board before confirming."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Use only the speech settings and local speech backend from this YAML")
    parser.add_argument("--output", type=Path, help="Export a WAV instead of playing sound")
    parser.add_argument("--backend", choices=("say", "espeak_ng", "piper", "gemini"), help="Explicit engine override")
    parser.add_argument("--sample", choices=("neutral", "chess"), default="neutral", help="Fixed audition text, never a live camera or microphone")
    parser.add_argument("--piper-model", type=Path, help="Existing local ONNX voice and matching .onnx.json; requires Piper")
    parser.add_argument("--piper-python", type=Path, help="Existing isolated speech interpreter; requires Piper")
    parser.add_argument("--gemini-voice", choices=("Charon", "Kore"), help="Prebuilt Gemini audition voice; requires Gemini")
    parser.add_argument("--allow-cloud-speech", action="store_true", help="Explicitly allow sending the fixed sample text to Gemini TTS")
    parser.add_argument("--repeat", type=int, default=1, help="1..5 explicit repetitions; repeated Piper uses one warm worker")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = load_settings(args.config)
        backend = args.backend or settings.models.text_to_speech_backend
        if backend not in (None, "espeak_ng", "say", "piper", "gemini"):
            raise ConfigurationError("Voice preview requires a supported local or explicitly enabled Gemini speech engine.")
        if not 1 <= args.repeat <= 5:
            raise ConfigurationError("Voice preview repeat count must be between 1 and 5.")
        if args.output is not None and args.repeat != 1:
            raise ConfigurationError("WAV export supports one sample at a time.")
        if backend == "gemini" and not args.allow_cloud_speech:
            raise ConfigurationError("Gemini voice preview requires --allow-cloud-speech; no sample was sent.")
        if args.allow_cloud_speech and backend != "gemini":
            raise ConfigurationError("--allow-cloud-speech requires an explicitly selected Gemini backend.")
        if args.gemini_voice is not None and backend != "gemini":
            raise ConfigurationError("--gemini-voice requires the Gemini speech backend.")
        if (args.piper_model is not None or args.piper_python is not None) and backend != "piper":
            raise ConfigurationError("Piper model/runtime overrides require the Piper backend.")
        if args.output is not None and args.output.suffix.lower() != ".wav":
            raise ConfigurationError("Voice preview output must use a .wav extension.")
        changes = {}
        if args.piper_model is not None:
            changes["piper_model_path"] = args.piper_model.expanduser().resolve()
        if args.piper_python is not None:
            changes["piper_python_path"] = args.piper_python.expanduser().absolute()
        if args.gemini_voice is not None:
            changes["gemini_voice"] = args.gemini_voice
        speech_settings = replace(settings.speech, **changes)
        sample = SAMPLE_TEXT
        if args.sample == "chess":
            from vidi.audio.speech_factory import render_chess_speech
            sample = render_chess_speech(CHESS_SAMPLE_TEXT)
        contexts = nullcontext()
        if backend == "gemini":
            from vidi.core.credentials import saved_gemini_key
            contexts = saved_gemini_key()
        speaker = None
        try:
            with contexts:
                if backend == "gemini" or (backend == "piper" and args.repeat > 1):
                    from vidi.audio.speech_factory import create_speech
                    speaker = create_speech(speech_settings, backend, cloud_enabled=args.allow_cloud_speech)
                else:
                    speaker = LocalSpeech(speech_settings, backend)
                for _ in range(args.repeat):
                    result = speaker.speak(sample, output_path=args.output.expanduser().resolve() if args.output is not None else None)
                    if result.status != "completed":
                        logging.error("Voice preview failed: %s", result.error or "Speech unavailable.")
                        return 130 if result.status == "cancelled" else 1
                    logging.info("Voice sample completed with %s. %s", result.backend,
                                 "WAV export only." if args.output is not None else "Confirm audibility through your selected audio output.")
        finally:
            if speaker is not None and hasattr(speaker, "close"):
                speaker.close()
        return 0
    except (ConfigurationError, CredentialError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.info("Voice preview stopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
