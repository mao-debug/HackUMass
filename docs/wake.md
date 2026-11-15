# Local hey vidi wake input

The optional assistant listens locally for **hey vidi** (intended pronunciation:
“hey VEE dee”). It uses sherpa-onnx 2.0.1 with the English GigaSpeech 3.3M
keyword-spotting model. The model accepts a tokenized custom phrase; it does not
require a cloud account or separate custom-model training. Runtime and model
licensing are Apache 2.0; the downloaded model's `README.md` carries its license
declaration. See the [official model instructions](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html#sherpa-onnx-kws-zipformer-gigaspeech-3-3m-2024-01-01-english),
[runtime license](https://github.com/k2-fsa/sherpa-onnx/blob/v2.0.1/LICENSE), and
[maintainer model-license discussion](https://github.com/k2-fsa/sherpa-onnx/issues/3802).

Idle microphone PCM stays in bounded RAM buffers and is never written to disk or
uploaded by wake detection. After an explicit wake, the command clip is a
canonical 16 kHz mono PCM16 WAV of at most five seconds, including up to 0.3
seconds of pre-roll within that limit. Only the explicitly enabled assistant
transcription action can send that clip to Gemini. Follow-up requests require
another wake. Privacy pause closes the microphone process and clears buffered
audio; resume starts a new capture session. Detection remains active during TTS
so a wake can cancel speech. There is no acoustic echo cancellation: full-duplex
Bluetooth operation, false activations from playback, and real interruption
latency still need the Pi/headset acceptance test. Synthesized feedback should
avoid saying the wake phrase itself.

## Approved Pi installation

Run from the checkout using the existing Python 3.11 `--system-site-packages`
`.venv-pi`. The user approved these PyPI packages and the official model release.
These commands install neither desktop requirements nor a replacement NumPy or
OpenCV. `--only-binary` prevents a surprise source build; `--no-deps` preserves
the Pi's OS libraries. The two runtime packages depend only on each other in
their published metadata. The second command supplies the tokenizer's imports;
those extras are unnecessary at runtime once `keywords.txt` exists.

```sh
cd ~/vidi
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: --no-deps sherpa-onnx==2.0.1 sherpa-onnx-core==2.0.1
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: --no-deps sentencepiece==0.2.3 pypinyin==0.55.0 click==8.5.1
```

Binary-wheel metadata resolution succeeded for Linux ARM64 Python 3.11 from
the Mac; installation/imports have not yet been tested on the physical Pi.
Do not replace the OS NumPy if a Pi command fails. The exact optional pins are
also recorded in [requirements-wake.txt](../requirements-wake.txt).

## Download and prepare the phrase

The approved source is this exact [official GitHub release asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2).
Download it into the ignored `models/` directory:

```sh
mkdir -p models/wake
curl --fail --location --output models/wake/gigaspeech-3.3M.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
sha256sum models/wake/gigaspeech-3.3M.tar.bz2
```

The archive downloaded and inspected on the Mac was **17,626,723 bytes** with
SHA-256:

```text
f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a
```

Compare the printed hash with that value before extracting. A mismatch needs
investigation; do not continue using a different archive.

```sh
tar -xjf models/wake/gigaspeech-3.3M.tar.bz2 -C models/wake
PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/prepare_wake.py --smoke
```

The script uses sherpa's [documented BPE tokenizer implementation](https://github.com/k2-fsa/sherpa-onnx/blob/v2.0.1/sherpa-onnx/python/sherpa_onnx/utils.py)
with the downloaded `bpe.model` and `tokens.txt`; it writes
`models/wake/keywords.txt`. It never downloads files, records audio, reads API
keys, or calls a cloud service. Do not type guessed tokens or substitute the
archive's example keywords. The approved pronunciation is encoded as English
letter names `HEY V D` using that tokenizer. Literal `HEY VIDI` failed local
spoken probes; the letter-name encoding recognized the same intended VEE-dee
sound at the original score/threshold. The generated line is `▁HE Y ▁ V ▁ D`;
tokenizer output alone is not a pronunciation test. Keep the model's README and
license attribution when copying the model directory. Model weights are not
committed to Git.

The adapter loads the int8 encoder, decoder and joiner with `chunk-16-left-64`
in their filenames, plus `tokens.txt`. `prepare_wake.py --smoke` loads those
files and feeds five seconds of synthetic zero PCM; it does not establish real
speech detection accuracy. Custom paths are available through `--model-dir`
and `--output`; generated keywords must remain outside the model directory.

## Verified locally and still pending

On macOS ARM64 Python 3.14.6, the approved binary wheels installed in the
existing isolated `.venv`: sherpa-onnx/core 2.0.1, sentencepiece 0.2.3,
pypinyin 0.55.0 and click 8.5.1. NumPy and OpenCV were not replaced. The real
tokenizer produced the phrase, the real model loaded, and five seconds of
synthetic silence produced zero wake events. No real microphone or cloud API
was used. Local provenance, archive/file hashes and notices are under ignored
`models/wake/manifest.json`, `NOTICE.txt`, and `SHERPA_ONNX_LICENSE`.

If you prepared keywords before the pronunciation fix, stop vidi, update the
feature branch, and rerun `scripts/prepare_wake.py --smoke` to replace the local
keyword file. Restart the assistant afterward; its loaded model does not
automatically reread a changed file.

Before enabling cloud commands, test the phrase with the headset locally:

```sh
PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/check_wake.py --seconds 15
```

Say “hey VEE dee,” leave a short pause, and repeat several times. Each hit
prints `Wake detected`; the final JSON reports input levels, processed audio,
and hit count. No camera, Gemini client, key, recording file or upload is used.
If there are no hits, retain the output rather than lowering sensitivity
without an input/false-trigger check. The camera application must be stopped
during this isolated test. A low or zero hit count is not a successful wake test.

The Pi test must still confirm “hey VEE dee” recognition, near-miss/ambient
false activations, CPU/load behavior alongside the camera, wake during actual
TTS playback, prompt cancellation, privacy pause/resume, and no delayed command
after cancellation. The input queue fails closed if inference falls behind
and drops audio. Neither the Mac smoke nor binary-wheel availability proves
Bluetooth full-duplex behavior, Pi real-time performance, or environmental
awareness reliability.
