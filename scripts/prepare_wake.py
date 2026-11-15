#!/usr/bin/env python3
"""Prepare the approved hey vidi keyword locally; never download or open audio."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from vidi.audio.wake_word import WakeDetector, WakeWordError, tokenize_keyword


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,
                        default=ROOT / "models/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")
    parser.add_argument("--output",type=Path,default=ROOT / "models/wake/keywords.txt")
    parser.add_argument("--smoke",action="store_true",help="also load the model and feed five seconds of synthetic silence")
    args = parser.parse_args(argv)
    temporary = None
    detector = None
    try:
        model, output = args.model_dir.resolve(), args.output.resolve()
        if output == model or output.is_relative_to(model):
            raise WakeWordError("Keep the generated keyword file outside the downloaded model directory.")
        line = tokenize_keyword(model)
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=output.parent,delete=False) as destination:
            temporary = Path(destination.name)
            destination.write(line)
        os.replace(temporary,output)
        temporary = None
        print("Prepared the local hey vidi keyword file:", output)
        if args.smoke:
            detector = WakeDetector(model,output)
            detector.prepare()
            hits = sum(detector.accept(b"\0"*640) for _ in range(250))
            if hits:
                raise WakeWordError("The silence smoke check unexpectedly detected a wake phrase.")
            print("Passed model loading and five seconds of synthetic silence; no microphone or cloud was used.")
        return 0
    except WakeWordError as error:
        print(str(error),file=sys.stderr)
        return 1
    except OSError:
        print("The local keyword file could not be prepared. Check folder permissions.",file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
