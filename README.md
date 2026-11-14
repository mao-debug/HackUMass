# vidi

vidi is a HackUMass smart glasses prototype for blind and visually impaired
users. The connected Pi 4 camera reports **IMX708 (Camera Module 3)**. Phase 2
uses Picamera2 capture and local OpenCV DNN CPU detection with MobileNet-SSD.
The separate IMX500 AI Camera backend remains optional for that different
sensor. macOS supports fixture replay. Explicit one-shot Gemini object detection
and scene descriptions are optional cloud demonstrations. Phase 3 adds local OCR
and a short speech response or WAV export. Voice commands remain deferred.
Pi OCR and audio acceptance are pending; the audio output device is unverified.

The default run processes one synthetic sample frame with detection disabled.
It opens no camera or preview, loads no model, makes no network request, and
produces no audio. This prototype supplements established mobility aids; it
does not measure precise distance or guarantee navigation safety.

## macOS and VS Code setup

Use Python 3.10 or newer. Open this repository in VS Code and activate a local
virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

After approval for any required package downloads:

```sh
python -m pip install -r requirements-dev.txt
```

In VS Code, select `.venv/bin/python` with **Python: Select Interpreter**.
`requirements.txt` installs the editable `desktop` extra: NumPy, OpenCV, and
PyYAML. `requirements-dev.txt` also installs Pytest. Raspberry Pi packages are
excluded. Existing local dependencies can be reused without another download.
After setup, sample/image processing and fixture replay work offline.

macOS is the fixture-replay development environment for Phase 2. The approved
CPU model can also be loaded against a sample/image for a setup smoke test;
this does not establish accuracy or Pi performance.

## Run locally

```sh
# Default: one synthetic frame, no detections or network.
python -m vidi

# Explicit default YAML configuration.
python -m vidi --config config/settings.yaml

# Replay invented detections; these are not model predictions.
python -m vidi --config config/fixtures.yaml --output artifacts/fixture.png --report artifacts/fixture.json

# Local image input without inference.
python -m vidi --image artifacts/fixture.png --detector none

# Optional webcam input and desktop preview without inference.
python -m vidi --source webcam --detector none --camera-index 0 --max-frames 60 --preview

python -m vidi --help
python -m vidi --version
```

The installed `vidi` command is equivalent to `python -m vidi`. Without
`--config`, built-in defaults match `config/settings.yaml` and work from any
directory after installation. CLI paths are relative to the current directory;
paths in YAML are relative to the YAML file's directory. CLI values override
the corresponding settings.

`--source` selects `sample`, `image`, `webcam`, `picamera2`, or `imx500`; `--image PATH`
selects image input and `--camera-index N` selects webcam input unless a source
is explicit. `--detector` selects `none`, `fixture`, `opencv_ssd`, `imx500`, or
`gemini`; the cloud backend also requires the explicit `--gemini-detect` flag.
`--detections-fixture PATH` supplies replay data. CPU detection uses `--model`
for the MobileNet-SSD `.caffemodel` and `--model-config` for `deploy.prototxt`.
It supports the model's 20 VOC object classes, including people, chairs, and
some vehicles; doors and signs are not supported classes. The optional IMX500
decoder instead accepts its specific baseline `.rpk` and model-aligned
`--labels`. Other networks require matching decoders. Use
`--threshold` for local-model confidence filtering and `--max-detections` to
limit results. Gemini labels have no SSD class ID or comparable confidence score.

`--max-frames` sets a positive frame limit. `--preview`/`--no-preview` control
the desktop window. **q** or **Escape** finishes capture; requested reading/cloud
actions then process the final frame. **Ctrl+C** cancels the run. `--output PATH`
saves the final frame, with overlays when detection is enabled, while
`--report PATH` writes a JSON result.
Both are explicit local exports. Generated files belong in ignored
`artifacts/`; use `.png` for an image export. `--log-level DEBUG` enables more
logging. A fixture report is explicitly marked simulated.

## Raspberry Pi setup and acceptance

The confirmed Pi 4 runs Debian 12, aarch64, Python 3.11.2, NumPy 1.24.2,
PyYAML 6, setuptools 66, Picamera2 0.3.27, and installed OS OpenCV 4.6.0. The
partner's monitor and keyboard are connected. **Preserve this installation.**
The Pi 5 with 8 GB RAM is requested and unverified. The active camera's IMX708
sensor is not IMX500; do not use an IMX500 model with it.

Supplied Pi logs confirm two camera-only runs and a 20-frame live CPU detection
run at 640×480. All 20 live frames were inference-ready; the final frame reported
`person` on a visible hand with model score 0.654. In a follow-up controlled test,
the user confirmed that a person box covered the subject's body. Broader object
checks and longer offline acceptance remain pending; see the
[bench record](docs/hardware.md).

Work from the repository root after the existing GitHub branch handoff or a
USB source transfer; no SSH setup is required. With the confirmed system
libraries, the first camera-only check needs no pip installation or model:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi.yaml --detector none --max-frames 3 --output artifacts/pi-camera.png
```

For the installed workflow, reuse system packages in a separate environment:

```sh
/usr/bin/python3 -m venv --system-site-packages .venv-pi
export PYTHONNOUSERSITE=1
.venv-pi/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
```

Do not install desktop requirements or pip NumPy/OpenCV/Picamera2 on the Pi.
Model downloads from the fixed author-repository commit have been approved;
see [docs/models.md](docs/models.md) for the exact URLs, MIT license, local
paths, and checksums. The application never downloads a model at startup.
Once those files are present, run a local load/forward smoke test, then a short
live test:

```sh
.venv-pi/bin/python -m vidi --source sample --detector opencv_ssd --model models/mobilenet-ssd/mobilenet_iter_73000.caffemodel --model-config models/mobilenet-ssd/deploy.prototxt --report artifacts/cpu-smoke.json
.venv-pi/bin/python -m vidi --config config/pi.yaml --max-frames 20 --no-preview --output artifacts/pi-detections.png --report artifacts/pi-cpu.json
```

`config/pi.yaml` requests 640×480 at a 2 FPS loop limit. The supplied 20-frame
Pi report shows about 2.00 FPS between capture timestamps and 13.26 seconds for
the full processing loop (about 1.51 frames/s including the initial frame wait).
These are different timing windows, not an isolated model-speed or accuracy
measurement. The later 100-frame offline bench procedure is in
[docs/phase2.md](docs/phase2.md). Missing local build prerequisites should be reported
before adding packages; the source-tree command above remains available.

## Read text and optional speech

`--read-text` runs local Tesseract once on the final **unannotated** frame,
after camera resources close. It requires detector `none` and cannot be
combined with a Gemini action. Add either `--speak` for one speech response or
`--speech-output FILE.wav` for a local WAV export. In this OCR workflow, pair
the output action with `--read-text`; speech is also available for the Gemini
actions below. Playback and WAV export cannot be combined. YAML settings alone
enable none of these actions.

The Pi OCR profile requests one 1280×960 frame, a one-second settling interval,
and a supported single autofocus cycle with a four-second wait limit. Camera
startup/capture remain bounded at 30/five seconds. Autofocus does not guarantee
readable text. After the pending OCR/speech package installation is approved
and completed, run from the repository root:

```sh
# Read locally without playing audio.
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi-ocr.yaml --read-text --output artifacts/pi-ocr.png --report artifacts/pi-ocr.json

# Optional WAV synthesis without playback.
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi-ocr.yaml --read-text --speech-output artifacts/pi-reading.wav --report artifacts/pi-reading.json

# One requested spoken response, once the output device/volume are verified.
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi-ocr.yaml --read-text --speak --report artifacts/pi-speaking.json
```

The default OCR language is English (`eng`), with grayscale preprocessing and
a 15-second process timeout. Speech uses 150 words per minute, a 60-word text
excerpt, and a 45-second timeout. Truncated excerpts add “More text is
available.” Empty OCR results use “No readable text found.” Full recognized
text remains available in an explicitly requested JSON report. Speech status
`completed` means the engine finished; it does not verify audibility.

The Mac has native `say`, but Tesseract is not installed. Approval is pending
for Homebrew Tesseract on Mac and the Pi OS packages `tesseract-ocr`,
`tesseract-ocr-eng`, and `espeak-ng`. No Python OCR/TTS wrapper is needed.
After setup, a local-image Mac run uses the existing desktop environment:

```sh
python -m vidi --image path/to/printed-text.png --detector none --read-text --report artifacts/mac-ocr.json
python -m vidi --image path/to/printed-text.png --detector none --read-text --speak
```

Mac live speech defaults to the native Samantha voice; the Pi profile uses
eSpeak NG's English voice. WAV export uses eSpeak NG, not native Mac `say`.
Package approval, setup, failure handling, and controlled bench checks are
documented in [docs/phase3.md](docs/phase3.md). The later 100-frame detection
and broader chair tests are separate checks and need not block trying OCR.

## Optional Gemini detection and descriptions

`--gemini-detect` explicitly requests one cloud object-detection result and
selects the `gemini` backend. `config/pi-gemini.yaml` prepares one 1280×960
Picamera2 snapshot; no local weights or Tesseract are needed. Selecting the
backend in YAML alone fails until the upload flag is supplied. After the
camera closes, only the final **unannotated** frame is sent, even when the
capture run requests multiple frames. This is not a per-frame cloud loop.

Gemini returns generated labels and validated boxes, converted to original
image pixels. Its `class_id` and `confidence` are `null`; local SSD confidence
thresholds do not apply. Broader label vocabulary does not guarantee correct
or complete coverage. Explicit `--threshold`, `--model`, and `--model-config`
options are rejected for Gemini rather than silently ignored. The separate
`--describe` action returns a short scene
description and accepts an optional `--question`. Choose one cloud action per
run. Optional `--speak` or `--speech-output FILE.wav` uses local speech output
for the resulting text; it does not call a Gemini speech-generation service.

Both actions require `GEMINI_API_KEY` and the pinned Google SDK
`google-genai==2.34.0`. The configured model is `gemini-3.5-flash-lite`, with a
15-second request timeout and a maximum 1024-pixel image edge. Output limits
are 1024 tokens for object detection and 256 for descriptions; the default
object limit is 10. No automatic retries or local-backend fallback are used.
With `--output` and `--report`, an API failure preserves the local image and
records the cloud result as unavailable. `config/pi.yaml` remains the separate
offline CPU detection path.

The SDK is approved and installed on the Mac; its existing compatibility
checks used mocked transport, not a live Gemini request. Approval now also
covers the pinned Pi SDK and required dependencies from PyPI in `.venv-pi`,
plus user-triggered non-sensitive test snapshots through a verified free-tier
project. Pi SDK installation, local key setup, and a live request have not yet
been performed. Keep the budget at **$0 new spending**: verify the intended
project's free tier, billing-disabled state, and model access before a test.
No billing changes, subscription credits, or student credits are assumed.

[docs/gemini.md](docs/gemini.md) explains AI Studio key creation, hidden Bash
entry, the approved Pi SDK setup, and the one-shot test commands.
Never paste a key into chat or commit it to GitHub. `.env` is not loaded;
[.env.example](.env.example) is a template only. Review the current
[Google data-use terms](https://ai.google.dev/gemini-api/terms) before an
approved non-sensitive image upload.

## Configuration, tests, and diagnostics

[config/settings.yaml](config/settings.yaml) is the safe default;
[config/fixtures.yaml](config/fixtures.yaml) selects replay, and
[config/pi.yaml](config/pi.yaml) selects IMX708/Picamera2 with CPU detection;
[config/pi-ocr.yaml](config/pi-ocr.yaml) selects the camera-only OCR snapshot,
[config/pi-gemini.yaml](config/pi-gemini.yaml) selects the opt-in cloud snapshot,
and [config/pi-imx500.yaml](config/pi-imx500.yaml) is the optional AI Camera profile.
The `camera`,
`simulation`, `detection`, `models`, `ocr`, `speech`, and `cloud` sections configure their
respective boundaries. Unknown keys and invalid values fail clearly. The
`models.object_detection_backend`, `object_detection_path`, and
`object_detection_config_path` settings select local detection assets.
`models.ocr_backend` and `models.text_to_speech_backend` select engines only
when the corresponding CLI action is requested.

```sh
python -m pytest
python scripts/check_hardware.py
python scripts/check_hardware.py --json
```

The hardware diagnostic is standard-library-only and reads local metadata,
module origins, OS identity, and possible camera device paths without importing
optional packages, opening devices, recording audio, or accessing the network.
Distribution metadata may be absent for an apt-provided module. Discovery alone
does not verify native ABI compatibility, camera capture, inference, or audio.
Camera capture and live CPU inference are recorded separately in
[docs/hardware.md](docs/hardware.md); broader detection validation and audio
acceptance remain pending. Phase 3 OCR/audio hardware checks have not yet run.
Tests use fixtures and mocked hardware/cloud
boundaries; they do not replace Pi acceptance tests.

Local validation used macOS arm64 and Python 3.14.6. Fixture image/report export,
the default CLI, and the installed Google SDK's request serialization were
checked. A spawned fake worker also passed a real Pipe/shared-memory transport
test outside the restricted filesystem sandbox, without accessing a camera.
The same test explicitly skips when the OS sandbox blocks shared memory.

If a camera is unavailable, use `--source sample --detector none` or fixture
replay. For a missing image, check path resolution. On a headless machine, use
`--no-preview`; on macOS, grant webcam access to VS Code or the terminal in
camera privacy settings when using that optional source. Missing Picamera2,
models, labels, or network connectivity should produce an actionable message
without silently changing the selected backend.

## Project layout and scope

The repository root is the vidi project. Code lives in `src/vidi/`: `camera/`
owns capture, `ai/object_detection.py` owns detection results and replay,
`ai/cpu_detection.py` owns CPU detection,
`ai/text_recognition.py` owns local OCR, `audio/text_to_speech.py` owns optional
speech, `ai/scene_description.py` owns the cloud boundary, and `core/` owns
configuration and the bounded controller. `main.py` and `__main__.py` expose the
CLI. Speech recognition and the voice assistant remain later-phase
scaffolding. `config/`, `examples/`, `scripts/`, `tests/`, and `docs/` hold
configuration, fixtures, diagnostics, verification, and setup notes.

Phase 1 established the foundation. Phase 2 covers camera/detection plus the
explicit one-shot Gemini demonstrations described above. Phase 3 adds requested local
OCR and speech; its installation and hardware acceptance remain pending.
Voice commands and wearable deployment/optimization remain later work.
Model class coverage is limited to the selected model's labels; an empty
detection list is not evidence that a path is clear. The full plan, official
references, approval gates, and hardware acceptance record are in
[docs/phase2.md](docs/phase2.md).
