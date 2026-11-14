# Optional local Piper voice

vidi can use Piper for one local spoken response or WAV export. **The choice
between Piper and Gemini speech, and approval to download Piper packages and
voice assets, are still pending.** Only documentation/catalog research has
been approved. Existing profiles keep eSpeak NG; nothing downloads at startup.
Piper voice quality, latency, and headphone playback have not been benchmarked
on the Pi.

The optional adapter runs Piper in a separate `.venv-tts` process. The camera,
OpenCV, and Google SDK stay in the existing `.venv-pi`. Once installed with a
local model, Piper synthesis needs no network or speech-generation API.
Gemini OCR remains a separate, explicit image upload when `--gemini-read` is
used; this integration does not call Gemini text-to-speech.

## Proposed voice and runtime

The candidate is `en_US-ljspeech-medium`: one US English female voice. Its
[author model card and pinned commit](https://huggingface.co/rhasspy/piper-voices/commit/bae641dcb5a608cff81ccaf7ff018baaca1a33e1)
describe the LJ Speech dataset as public domain. The ONNX model is
63,531,379 bytes, about 64 MB; its matching `.onnx.json` is also required.
Keep the accompanying `MODEL_CARD` with the assets.

`requirements-tts.txt` pins `piper-tts==1.8.0`; its dependencies are installed
inside `.venv-tts`. [PyPI release metadata](https://pypi.org/project/piper-tts/1.8.0/)
lists Python 3.9 or newer, Linux ARM64 and macOS wheels, and the engine's
GPL-3.0-or-later license. The intended Pi interpreter is its existing Python
3.11.2. Wheel availability is not an on-device performance result. The
[Piper project](https://github.com/OHF-Voice/piper1-gpl) and
[CLI documentation](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md)
describe local synthesis; the CLI loads the voice for each invocation.

## Install only after explicit approval

**Do not run this section until package/dependency and voice-download approval
is given.** These are proposed Pi steps from the repository root. They contact
PyPI and Hugging Face; they have not been executed as part of this integration.

Create an isolated environment without `--system-site-packages`; keep the
working `.venv-pi` intact:

```sh
/usr/bin/python3 -m venv .venv-tts
.venv-tts/bin/python -m pip install --only-binary=:all: -r requirements-tts.txt
```

If no compatible wheel is available, stop and report the error before changing
the runtime or attempting a source build. No audio-server installation is
part of these steps.

Download the model, matching configuration, and model card at fixed commit
`bae641dcb5a608cff81ccaf7ff018baaca1a33e1`:

```sh
mkdir -p models/piper
curl --fail --location --output models/piper/en_US-ljspeech-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/bae641dcb5a608cff81ccaf7ff018baaca1a33e1/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx
curl --fail --location --output models/piper/en_US-ljspeech-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/bae641dcb5a608cff81ccaf7ff018baaca1a33e1/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json
curl --fail --location --output models/piper/MODEL_CARD https://huggingface.co/rhasspy/piper-voices/resolve/bae641dcb5a608cff81ccaf7ff018baaca1a33e1/en/en_US/ljspeech/medium/MODEL_CARD
sha256sum models/piper/en_US-ljspeech-medium.onnx
```

The expected ONNX SHA-256, recorded in the author's Git LFS pointer, is:

```text
6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22
```

Compare it exactly before use; stop if it differs. On macOS, the equivalent
checksum command is `shasum -a 256 models/piper/en_US-ljspeech-medium.onnx`.
The application requires existing local assets and never fetches missing ones.

## Preview without a camera or cloud request

After approved setup, pair the headphones and select them as the desktop audio
output as described in [Phase 3](phase3.md#bluetooth-headphones-on-the-pi).
The preview script speaks a fixed short sample using only the selected local
speech settings. It does not open a camera, read a saved API key, or call
Gemini, even when its configuration also contains OCR settings.

Compare the existing eSpeak NG voice and optional Piper voice at a comfortable
volume:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --config config/pi-gemini-ocr.yaml
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --config config/pi-piper-ocr.yaml
```

The eSpeak comparison requires its installed executable. To generate a Piper
sample without playing it, use:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/preview_voice.py --config config/pi-piper-ocr.yaml --output artifacts/piper-preview.wav
```

A completed engine run does not establish physical audibility. Missing runtime,
model/configuration, player, or invalid WAV output produces a failure rather
than a download or automatic engine fallback. Ctrl+C stops the local work.

## Select Piper for one reading

After the user chooses this voice and setup/playback are verified, stop any
live camera command before taking the separate OCR snapshot. This command
uploads one newly captured image to Gemini, preserves the full transcription,
and requests one local Piper excerpt:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-piper-ocr.yaml --gemini-read --speak --report artifacts/pi-piper-reading.json
```

Replace `--speak` with `--speech-output artifacts/pi-piper-reading.wav` for
WAV export without playback. Choosing the original `config/pi-gemini-ocr.yaml`
profile keeps eSpeak NG. Neither profile activates speech without an explicit
speech flag.

The Piper profile sets these speech values; paths in YAML resolve relative to
the YAML file:

| Setting | Value or behavior |
| --- | --- |
| `models.text_to_speech_backend` | `piper` |
| `speech.piper_python_path` | `../.venv-tts/bin/python` |
| `speech.piper_model_path` | `../models/piper/en_US-ljspeech-medium.onnx` |
| Matching voice configuration | Same path plus `.json` |
| `speech.piper_length_scale` | `1.1`; duration scale, not words per minute |
| `speech.max_words` | 60 recognized words, then truncation feedback |
| `speech.timeout_s` | 60 seconds shared by synthesis, validation, and playback |
| `speech.player` | Omitted/null: automatic local player selection |

Linux automatic playback selects an installed `paplay`, then `pw-play` if
`paplay` is absent. macOS selects `afplay`. `aplay` is available only through
explicit `speech.player: aplay`; it is not an automatic Bluetooth fallback.
An explicit player must be installed. A selected player failure is reported
without retrying another player or changing audio routes. WAV export needs no
player and never plays sound.

OCR is checkpointed before speech. The complete text remains in the report if
speech fails or is interrupted; only the speech copy is limited to 60 words.
Check intelligibility, startup time, cancellation, WAV export, and actual
headphone playback on the Pi before selecting this voice for the demonstration.
