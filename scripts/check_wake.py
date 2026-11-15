#!/usr/bin/env python3
"""Check real local wake detection and input levels without cloud or recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidi.audio.continuous_input import AudioInput
from vidi.audio.speech_to_text import SpeechRecognitionError
from vidi.audio.wake_word import WakeDetector, WakeWordError
from vidi.core.configuration import ConfigurationError, load_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/pi-phase6.yaml")
    parser.add_argument("--seconds", type=int, default=15, help="Local listening window, 1–30 seconds")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
    audio = detector = None
    try:
        settings = load_settings(args.config)
        assets = settings.assistant
        if assets.wake_model_dir_path is None or assets.wake_keywords_path is None:
            raise ConfigurationError("Configure local wake model and keyword paths first.")
        detector = WakeDetector(assets.wake_model_dir_path, assets.wake_keywords_path,
                                threshold=assets.wake_threshold)
        detector.prepare()
        audio = AudioInput(settings.voice.device)
        audio.start()
        import numpy as np

        print("Local wake check active. Say hey VEE dee, pause, then repeat. No audio is saved or uploaded.", flush=True)
        started = time.monotonic()
        count = byte_count = 0
        peak = 0
        squares = 0.
        while time.monotonic() - started < args.seconds:
            pcm = audio.read(timeout_s=.1)
            if audio.dropped_bytes:
                raise SpeechRecognitionError("Wake processing fell behind and dropped input. Stop the camera and retry this check.")
            if not pcm:
                continue
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
            byte_count += len(pcm)
            peak = max(peak, int(np.max(np.abs(samples))))
            squares += float(np.dot(samples, samples))
            if detector.accept(pcm):
                count += 1
                print(f"Wake detected: {count}", flush=True)
        sample_count = byte_count // 2
        print(json.dumps({"wake_count": count, "audio_seconds": byte_count / 32000,
                          "peak_percent": round(100 * peak / 32768, 2),
                          "rms_percent": round(100 * (squares / sample_count) ** .5 / 32768, 2) if sample_count else 0.,
                          "dropped_bytes": audio.dropped_bytes, "cloud_requests": 0,
                          "audio_saved": False}))
        if not byte_count:
            raise SpeechRecognitionError("No microphone samples arrived during the local wake check.")
        return 0
    except (ConfigurationError, SpeechRecognitionError, WakeWordError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, ImportError):
        print("Local wake check unavailable. Verify the approved packages, model files and headset input.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if audio is not None:
            audio.close()
        if detector is not None:
            detector.request_close()


if __name__ == "__main__":
    raise SystemExit(main())
