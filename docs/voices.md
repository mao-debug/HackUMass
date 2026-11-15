# Voice setup and audition

vidi supports the existing eSpeak NG/macOS `say` engines, optional local Piper,
and explicitly enabled Gemini speech. Selecting a voice does not activate a
microphone or camera. There is no automatic switch between providers.

The user has approved Piper 1.8.0 and its required PyPI dependencies, both
voices below from Hugging Face, and fixed audition/selected runtime text uploads
to Gemini TTS. Further packages, models, or services still need approval. Nothing
downloads at application startup. Keep the working camera environment intact.
The user selected LJ Speech after listening to the local samples. Phase 6
therefore uses local LJ Piper by default; the alternatives remain explicit
audition choices.

## Current verification

On the development Mac, Piper installed successfully in an isolated `.venv-tts`
using the existing Python 3.14.6 ARM64 interpreter and binary wheels. Both voices
generated valid 22,050 Hz mono 16-bit WAV files without playback. Each warm
session reused one child process for two utterances. Local evidence is in the
ignored `artifacts/piper-local-synthesis-checks.json`; package and asset manifests
are `artifacts/piper-mac-install.json` and `artifacts/piper-voice-manifest.json`.

The Gemini adapter was checked with the installed `google-genai==2.34.0` SDK
and an offline HTTP mock. No live Gemini speech request has been tested here.
Pi installation, Bluetooth playback with these new voices, simultaneous
listening/playback, and end-to-end interruption remain
on-device acceptance checks. Mac synthesis timings do not establish Pi latency.

## Approved local voices

| Voice | Character | Immutable source revision | ONNX SHA-256 |
| --- | --- | --- | --- |
| `en_US-ljspeech-medium` | US English female, 22,050 Hz | `bae641dcb5a608cff81ccaf7ff018baaca1a33e1` | `6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22` |
| `en_US-john-medium` | US English male, 22,050 Hz | `e70533d22ac02da609b7427dd1a8bd5349295e98` | `789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988` |

Each ONNX file is 63,531,379 bytes and needs its matching `.onnx.json`. Keep its
model card too. Both [LJ Speech](https://huggingface.co/rhasspy/piper-voices/blob/bae641dcb5a608cff81ccaf7ff018baaca1a33e1/en/en_US/ljspeech/medium/MODEL_CARD)
and [John](https://huggingface.co/rhasspy/piper-voices/blob/e70533d22ac02da609b7427dd1a8bd5349295e98/en/en_US/john/medium/MODEL_CARD)
cards identify public-domain training data. Piper's engine is
[GPL-3.0-or-later](https://github.com/OHF-Voice/piper1-gpl/tree/v1.8.0);
retain the respective license and model information when distributing assets.

From the repository root on the Pi, create a separate environment using its
existing Python. Do not use `--system-site-packages` for this speech environment:

```sh
/usr/bin/python3 -m venv .venv-tts
.venv-tts/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: -r requirements-tts.txt
```

The pinned [Piper release](https://pypi.org/project/piper-tts/1.8.0/) provides
the runtime; its dependencies include ONNX Runtime, NumPy, and pathvalidate.
If compatible wheels are unavailable, stop before changing Python or attempting
a source build. The Mac setup used ONNX Runtime 1.32.0; the Pi resolver must
select wheels compatible with its existing Python 3.11 ARM64 environment.

Download the approved fixed assets. These commands contact Hugging Face:

```sh
mkdir -p models/piper
curl --fail --location --output models/piper/en_US-ljspeech-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/bae641dcb5a608cff81ccaf7ff018baaca1a33e1/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx
curl --fail --location --output models/piper/en_US-ljspeech-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/bae641dcb5a608cff81ccaf7ff018baaca1a33e1/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json
curl --fail --location --output models/piper/en_US-ljspeech-medium.MODEL_CARD https://huggingface.co/rhasspy/piper-voices/resolve/bae641dcb5a608cff81ccaf7ff018baaca1a33e1/en/en_US/ljspeech/medium/MODEL_CARD
curl --fail --location --output models/piper/en_US-john-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/e70533d22ac02da609b7427dd1a8bd5349295e98/en/en_US/john/medium/en_US-john-medium.onnx
curl --fail --location --output models/piper/en_US-john-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/e70533d22ac02da609b7427dd1a8bd5349295e98/en/en_US/john/medium/en_US-john-medium.onnx.json
curl --fail --location --output models/piper/en_US-john-medium.MODEL_CARD https://huggingface.co/rhasspy/piper-voices/resolve/e70533d22ac02da609b7427dd1a8bd5349295e98/en/en_US/john/medium/MODEL_CARD
sha256sum models/piper/en_US-ljspeech-medium.onnx models/piper/en_US-john-medium.onnx
```

Compare the hashes exactly with the table before using the files. On macOS use
`shasum -a 256` instead of `sha256sum`. The Mac download manifest also records
the configuration/card hashes and fixed source URLs. No assets or virtual
environments belong in Git.

## Compare fixed samples

Pair/select the headphones using the [existing Bluetooth setup](phase3.md#bluetooth-headphones-on-the-pi).
Run each neutral and chess sample at a comfortable, consistent volume. The
script uses fixed text, never a camera frame or microphone recording. Local
auditions do not access a saved API key.

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --backend espeak_ng --sample chess
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --backend piper --piper-python .venv-tts/bin/python --piper-model models/piper/en_US-ljspeech-medium.onnx --sample chess --repeat 2
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --backend piper --piper-python .venv-tts/bin/python --piper-model models/piper/en_US-john-medium.onnx --sample chess --repeat 2
```

Omit `--sample chess` for the neutral sample. On Mac use `.venv/bin/python`;
use `--backend say` for the existing local baseline. `--repeat 2` reuses a warm
Piper worker; a single local audition uses the existing per-invocation adapter.
To export without any playback, replace `--repeat 2` with
`--output artifacts/piper-chess.wav`. WAV export allows one sample at a time.
Ctrl+C cancels the audition and closes its worker.

Piper configuration uses `models.text_to_speech_backend: piper`,
`speech.piper_python_path: ../.venv-tts/bin/python`, and
`speech.piper_model_path: ../models/piper/en_US-ljspeech-medium.onnx` in a YAML
file under `config/`. Paths are relative to that file. The existing
`config/pi-piper-ocr.yaml` remains available for a requested OCR reading.

The Phase 6 speech factory keeps a loaded Piper model in its isolated worker
for successful consecutive utterances. Cancelling inference destroys that
worker and discards late audio; a later request starts a new worker. Playback
is locally cancellable. No device routes or audio servers are changed.

## Explicit Gemini speech audition

The Gemini audition defaults to `gemini-3.8-flash-lite-tts`, with `Charon`; `Kore` is
the second audition candidate. The current official
[speech generation documentation](https://aistudio.google.com/docs/speech-generation)
describes Interactions streaming PCM at 24 kHz. The adapter uses the installed
Google SDK, one stateless request per utterance, no retries/history/tools, and
bounded text, audio, and timeout limits.

These commands send only the fixed chosen sample text to Google, using the
existing saved credential mechanism. They do not upload camera/microphone data:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --backend gemini --allow-cloud-speech --gemini-voice Charon --sample chess
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --backend gemini --allow-cloud-speech --gemini-voice Kore --sample chess
```

Without `--allow-cloud-speech`, Gemini audition fails before key access or a
request. Runtime Gemini speech likewise requires the explicitly selected
backend and runtime `--cloud-speech`/`--cloud` enablement; a key alone does not
authorize uploads. Configuration uses `speech.gemini_model`,
`speech.gemini_voice`, and an optional `speech.gemini_style`.

Linux can play streamed raw PCM through the existing `paplay`. Other players
and Mac receive a completed validated WAV before playback. Stop immediately
cancels local playback and discards late results; closing a cloud transport is
best effort and does not establish server-side cancellation or refunded usage.
Connection, quota, format, and player failures do not switch providers.
Transport cleanup is bounded locally; another Gemini utterance is refused
while prior cleanup remains unfinished rather than accumulating background work.
Consult the project's active quota in AI Studio and Google's current
[pricing](https://aistudio.google.com/docs/pricing#gemini-3.8-flash-lite-tts)
before use; the application makes no billing changes.

## Choose the demonstration voice

Ask the intended listener to compare neutral speech, chess coordinates,
captures, castling, and a longer printed-text excerpt. Record misunderstood
words and coordinates, comfort, first-audio delay, total duration, and stop
delay for each candidate. Compare cold and repeated warm Piper separately;
record network conditions for Gemini. A valid WAV does not prove audibility
or intelligibility.

On the Pi, verify the chosen voice through the actual headphones, stop speech
mid-sentence, then confirm the next command works without old audio. Test
offline behavior and a failed cloud request. Simultaneous wake listening and
Bluetooth playback need a separate check: microphone and speaker success in
separate tests does not establish full-duplex behavior. Keep eSpeak available
as an explicitly selected offline profile until the new choice passes.

Only speech rendering expands isolated chess coordinates (for example, `g1`
becomes “G one”). Stored moves, board state, and confirmation IDs are unchanged.
Chess speech reports observations and proposals; it does not choose moves.
