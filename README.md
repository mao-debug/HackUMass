# vidi

vidi is a HackUMass smart glasses prototype for blind and visually impaired
users. The current target is a **Raspberry Pi 5 using the previous Pi 4 SD
card**. The connected camera reports **IMX708 (Camera Module 3)**. Phase 2
uses Picamera2 capture and local OpenCV DNN CPU detection with MobileNet-SSD.
The separate IMX500 AI Camera backend remains optional for that different
sensor. macOS supports fixture replay. Explicit one-shot Gemini object detection
and scene descriptions are optional cloud demonstrations. Phase 3 adds Gemini
or local OCR and a short local speech response or WAV export. Gemini OCR and
audible speech have passed a user-run Pi 5 check.
Phase 4 adds requested scene descriptions and independent left/right ultrasonic
feedback over an ESP32 USB link; the user confirmed smooth combined operation
and audible feedback. Phase 5 adds requested short voice commands: the Pi
accepted a spoken scene-description command, and the user heard both scene
feedback and printed-text reading. Voice-triggered OCR is not confirmed by
the supplied log. Further measured hardware checks and
upgrades are deferred to Phase 6 at the user's request.

The [three-minute demo guide](docs/demo.md) provides a return-to-Pi setup,
presentation sequence, printed card, and offline fallback without claiming
unmeasured accuracy or performance. Deferred optimization and hardware work
are tracked in the [Phase 6 plan](docs/phase6.md).

The default run processes one synthetic sample frame with detection disabled.
It opens no camera or preview, loads no model, makes no network request, and
produces no audio. This prototype supplements established mobility aids; it
does not measure precise distance or guarantee navigation safety.

## Demo launcher

From the repository root, inspect a plan without starting anything:

```sh
python3 scripts/run_demo.py
```

The launcher defaults to **synthetic simulation and offline mode**. It only
prints the selected command/report path; `--run` is required to start the app.
For an away-from-Pi rehearsal with simulated proximity:

```sh
python3 scripts/run_demo.py --proximity --run
```

Offline mode explicitly blocks **D/T/M**, does not load saved credentials,
and keeps **H/P/S/R/Q** local; P requires `--proximity`. Add `--speak` for
requested native Mac speech. This uses synthetic pixels and fixture readings,
not Pi hardware. Reports get unique filenames under the repository's ignored
`artifacts/` directory. The launcher reuses `.venv/bin/python` for simulation
or `.venv-pi/bin/python` for `--target pi`; `--python PATH` overrides the existing
interpreter and `--artifacts-dir DIR` changes the report directory.

Cloud actions require explicit `--cloud`; M commands also require `--voice`.
Simulation voice needs `--command-audio FILE.wav`; Pi voice can optionally use
`--microphone-device NAME`. `--prompt-key` requires cloud mode and takes no key
value. The launcher installs nothing, changes no audio routes, and starts no
video recording. With `--run`, the app runs in the foreground with its usual
cancellation behavior.

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
`gemini`; the cloud backend also requires `--gemini-detect` for a snapshot or
`--continuous --gemini-live` for repeated images.
`--detections-fixture PATH` supplies replay data. CPU detection uses `--model`
for the MobileNet-SSD `.caffemodel` and `--model-config` for `deploy.prototxt`.
It supports the model's 20 VOC object classes, including people, chairs, and
some vehicles; doors and signs are not supported classes. The optional IMX500
decoder instead accepts its specific baseline `.rpk` and model-aligned
`--labels`. Other networks require matching decoders. Use
`--threshold` for local-model confidence filtering and `--max-detections` to
limit results. Gemini labels have no SSD class ID or comparable confidence score.

`--max-frames` sets a positive frame limit. `--continuous` instead keeps local
camera capture and detection running until you stop it; it ignores the YAML
frame limit and cannot be combined with `--max-frames`.
`--preview`/`--no-preview` control
the desktop window. **q** or **Escape** finishes capture; requested reading/cloud
actions then process the final frame. **Ctrl+C** cancels the run. `--output PATH`
saves the final frame, with overlays when detection is enabled, while
`--report PATH` writes a JSON result.
Both are explicit local exports. Generated files belong in ignored
`artifacts/`; use `.png` for an image export. `--log-level DEBUG` enables more
logging. A fixture report is explicitly marked simulated.

For a responsive continuous camera view with the approved repeated Gemini
identification, use the existing Pi environment and saved key:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-live.yaml --continuous --gemini-live --preview --report artifacts/pi-live-gemini.json
```

The left pane shows the current camera feed; the right shows the selected
image and request state, with boxes, frame ID, and age after a successful result.
Delayed boxes are never drawn onto a newer
frame. The profile requests 640×480 at **30 FPS**, while Gemini runs in one
background worker. The earlier Pi 4 run measured **11.21 FPS**; the later Pi 5
run measured **23.43 FPS** as recorded below. The requested
rate is not a throughput guarantee. Live Gemini uses a **60-second request
timeout**; snapshot requests keep their existing 15-second timeout.
`cloud.live_interval_s` is at least five seconds **after a response completes**;
there is no overlapping request or accumulated frame queue. A failed request
stops uploads and leaves the camera preview running. Restart the command to
try again. A pending request or timeout is shown explicitly; "No objects
identified" appears only after a successful empty response. A longer timeout
allows more response time and does not guarantee detection. `--gemini-live`
explicitly authorizes repeated images to Google;
the YAML profile or saved key alone cannot enable uploads. This mode uses the
existing approved free-tier project and makes no billing changes.

Press **q** or **Esc** in the preview, or **Ctrl+C** in the terminal, to stop.
No further frames are submitted; an already active request may finish. Final
exports, if requested, are written after stopping. An optional `--output`
image is the last analyzed frame with its matching boxes. The JSON report
distinguishes camera frames from received analyses and retains only the latest
120 analysis records plus full-run totals. This is a live feed, not a saved
video file. OCR and speech remain separate explicit snapshot actions.

For a camera-only speed comparison, use the same fast profile with
`--detector none --continuous --preview` and omit `--gemini-live`; it makes no
uploads or key-file reads. The original `config/pi.yaml` remains the reproducible
2 FPS local MobileNet-SSD baseline. That temporary model supports only 20 VOC
classes and can mislabel unsupported objects such as windows and computer
mice. It is not used by the Gemini live command above.

## Phase 4: requested scenes and local proximity

The existing ESP32 sketch now uses **protocol v1**: an identification handshake
and a newly acquired left/right pair for each request. **Upgrade the partner's
old streaming firmware before connecting this reader**; its fused center/TTC/
confidence/clearance output is deliberately rejected. Pins remain left
GPIO25/34 and right GPIO26/35. Verify the separate echo dividers and common
ground using [esp32code/README.md](esp32code/README.md). Native firmware tests
do not establish a physical sensor pass.

Only the optional USB reader needs `pyserial==3.5`. That download is approved
and installed on Mac. On the Pi, the approved isolated install is:

```sh
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m pip install --no-deps pyserial==3.5
PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/check_sensors.py --list
```

Port listing opens no device. Use the actual board path returned by discovery;
`/dev/ttyUSB0` below is an example, not automatic port selection. Close Arduino
Serial Monitor before opening the same port.

```sh
# Pi desktop: no upload until D; no speech unless --speak is added.
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase4.yaml --continuous --scene-live --preview --proximity --sensor-port /dev/ttyUSB0 --report artifacts/phase4.json

# One camera snapshot, one requested description, then fresh local readings/audio.
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase4.yaml --describe --proximity --sensor-port /dev/ttyUSB0 --speak --report artifacts/phase4-snapshot.json

# Mac: explicitly simulated camera/proximity; P is local, D requests Gemini.
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv/bin/python -m vidi --config config/phase4-sim.yaml --continuous --scene-live --preview --proximity --report artifacts/phase4-sim.json
```

In the focused scene preview, **D** requests one owned camera frame from
Gemini; **P** requests local proximity without an upload; **S** stops speech
and pending narration; **R** rearms failed cloud requests and reconnects the
sensor reader; **q/Esc** quits. `--speak` enables one short response per accepted
action. No automatic narration or request queue is started. D uses the existing
saved key/free-tier scope and waits at least five seconds after a completed
cloud response before another request. R alone does not upload.

Left/right are independent sensor channels, not object labels, a measured
center, or distances to objects named by Gemini. The provisional nearby
threshold is **1000 mm**, with **0.5-second** freshness expiry. Missing, stale,
out-of-range, or failed readings stay unknown; they never establish clearance.
Speech describes a snapshot checked when narration starts, not continuously
updated conditions throughout the utterance. Existing Gemini live detection
and OCR/read-aloud remain separate commands.

[docs/phase4.md](docs/phase4.md) contains setup, failure behavior, protocol,
simulation, and the deferred controlled sensor bench. Real paired samples and
combined camera/Gemini/sensor/audio operation were supplied; USB logs showed
unavailable readings on disconnect and recovery afterward. Distance accuracy,
resistor/echo voltage, supply quality, and sustained performance were not
validated and are deferred to Phase 6. Optional Piper downloads remain pending.

## Phase 5: requested voice commands

The new `--voice-commands` flag requires `--scene-live --continuous --preview`. **M**
records one clip of at most five seconds and sends it to Gemini transcription;
a local English allowlist selects describe, read text, proximity, stop, or help.
There is no continuous listening or wake word. The user approved these short
audio uploads using the existing SDK/key and free-tier scope; no new Python
packages are required. The user recorded a requested five-second clip through
the Pi's Bluetooth microphone and heard their recorded voice through playback.
The subsequent Pi run logged HTTP 200 and an accepted `describe` voice
command, followed by camera text and separate left/right sensor feedback.
The user heard both that description and a printed-text response. The log
does not demonstrate an accepted `read_text` voice command, so voice-triggered
OCR remains unconfirmed.

Select the headset as the desktop microphone input, then use the existing Pi
environment and observed ESP32 port:

```sh
python3 scripts/run_demo.py --target pi --cloud --voice --proximity --sensor-port /dev/ttyUSB0 --speak --run
```

This selects the Pi profile/interpreter and prints a unique report path under
ignored `artifacts/`. The manual `python -m vidi` entry point remains documented
in [Phase 5](docs/phase5.md).

In the focused window, **M** starts one command clip; **T** requests Gemini
printed-text reading; **H** gives local help. **D/P/S/R/q/Esc** keep their scene,
proximity, stop, rearm, and quit controls. Playback and help honor `--speak`.
Say “What's in front of me?”, “Read this sign,” “Check proximity,” “Stop
speaking,” or “Help” after M. A spoken stop requires recording/transcription;
use **S or q/Esc** for immediate local cancellation. Busy commands are not
queued. A recognized camera action waits for the shared five-second cloud
cooldown after transcription before making its separate image request.

Voice transcription and D/T require network access. Keyboard **P/H/S** remain
local; starting the window does not record or upload audio. Audio clips and raw
transcripts are not saved in reports; canonical actions/status/model/timing
are recorded alongside requested scene/OCR/proximity results.

For optional cloud voice replay on Mac, supply your own PCM16 WAV, mono/stereo,
8–48 kHz, at most five seconds:

```sh
python3 scripts/run_demo.py --cloud --voice --proximity --command-audio /path/to/your-command.wav --run
```

Replace the WAV path; each M sends that explicit clip for transcription.
Optional `--speak` uses native Mac `say`. The default Pi capture device is
literal ALSA `pulse`; `--microphone-device` can override it and cannot be paired
with WAV replay. Detailed input selection, command limits, failures, and
recorded functional evidence are in [docs/phase5.md](docs/phase5.md).
The dedicated higher-resolution Phase 3 OCR snapshot profile remains available.

## Raspberry Pi setup and acceptance

The user has moved the working Pi 4 SD card into a Pi 5. Start with the
[Pi 5 migration checks](docs/pi5-migration.md): identify the board/runtime,
enumerate the camera, test capture without AI, then recheck audio and the
existing application. The same `config/pi-*.yaml` profiles are shared between
the boards; no board-specific pipeline or `/dev/videoN` path is selected by
the application. The supplied report identifies **Pi 5 Model B Rev 1.1**, a
64-bit process, and **8061 MiB of kernel-reported memory**. It runs Bookworm,
kernel `6.12.115+rpt-rpi-2712`, and Python 3.11.2 inside the retained `.venv-pi`.
Picamera2 0.3.31 and Google SDK 2.34.0 are reported; OS OpenCV/NumPy/libcamera
modules and eSpeak NG are discoverable. `rpicam-hello` enumerated IMX708 at
index 0. A subsequent camera-only application run processed **282 frames at
21.79 FPS**, demonstrating working native camera/OpenCV/NumPy capture and
preview. This is the application delivery rate, not a sensor-speed guarantee.
That run printed nonfatal libpng/GLib messages. Gemini live identification then
processed **684 frames at 23.43 FPS**, with **four successful responses and zero
unavailable results**. The user confirmed hearing the Pi 5 eSpeak test word,
then completed Gemini reading with **OCR=ready, speech=completed** and heard the
spoken excerpt. The report was saved to `artifacts/pi5-gemini-speaking.json`.
Image colors/clarity, recognition accuracy, and power/thermals still need
confirmation. No package reinstallation is indicated by the supplied results.

The recorded **Pi 4 baseline** runs Debian 12, aarch64, Python 3.11.2, NumPy 1.24.2,
PyYAML 6, setuptools 66, Picamera2 0.3.27, and installed OS OpenCV 4.6.0. The
partner's monitor and keyboard are connected. **Preserve this installation.**
Those versions and measurements describe the previous board and must be
checked on the Pi 5. An IMX708 sensor is not IMX500; changing boards does not
change the camera model.

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

An optional **Piper neural voice** backend and fixed voice-preview script are
available for comparing a more natural local voice. Its engine/model downloads
still require approval, and Pi voice quality/latency have not been verified.
The isolated `.venv-tts` environment keeps speech dependencies separate from
the camera environment. See [voice setup and audition](docs/voices.md) before
using `config/pi-piper-ocr.yaml`. Existing profiles continue to select eSpeak NG.

Phase 3 also supports **Gemini printed-text reading** with your existing SDK
and saved API key. Stop the live camera command with q/Esc first, then run:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini-ocr.yaml --gemini-read --report artifacts/pi-gemini-reading.json
```

`--gemini-read` explicitly sends one final unannotated image to Google after
the camera closes. It transcribes printed words and line breaks rather than
describing the scene or answering instructions in the image. The profile
requests one 1280×960 frame with bounded settling/autofocus, uploads a JPEG
with at most a 1024-pixel longest edge, uses a 60-second OCR request timeout,
and permits 2048 output tokens. Start with a short, clear printed sign; small
print and long pages need controlled testing. There are no automatic retries
or local OCR fallback. A model transcription can contain errors.

Add `--speak` for one local spoken excerpt after the Bluetooth headphones are
paired, selected as the Pi output, and the approved `espeak-ng` package is
installed. `--speech-output FILE.wav` instead synthesizes a WAV without playing
it. Gemini reading needs no Tesseract installation or additional Python package.
Full text, model, status, and timing are saved before optional speech begins,
so speech failure does not discard the reading. The offline option remains:

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
readable text. After the pending Tesseract package installation is approved
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

The Mac has native `say`, but Tesseract is not installed. Pi `espeak-ng` and its
required dependencies are approved from the configured OS apt repositories;
the Pi 5 reports the engine installed, and the user heard both its test word
and a Gemini OCR excerpt. Approval is still
pending for Homebrew Tesseract on Mac and Pi `tesseract-ocr`/
`tesseract-ocr-eng`. No Python OCR/TTS wrapper is needed.
After setup, a local-image Mac run uses the existing desktop environment:

```sh
python -m vidi --image path/to/printed-text.png --detector none --read-text --report artifacts/mac-ocr.json
python -m vidi --image path/to/printed-text.png --detector none --read-text --speak
```

Mac live speech defaults to the native Samantha voice; the standard Pi profile
uses eSpeak NG's English voice. WAV export uses the selected eSpeak NG or
optional Piper backend; native Mac `say` supports playback only.
Package approval, setup, failure handling, and controlled bench checks are
documented in [docs/phase3.md](docs/phase3.md). The later 100-frame detection
and broader chair tests are separate checks and need not block trying OCR.

## Optional Gemini detection and descriptions

`--continuous --gemini-live` selects the asynchronous live workflow above.
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

Gemini actions require an API key and the pinned Google SDK
`google-genai==2.34.0`. Run `python -m vidi --save-key` once in the appropriate
environment to enter the key privately and save it on that device. It takes
no key argument, must be used by itself, and makes no network request. The key
is stored as **unencrypted plain text** in `~/.config/vidi/gemini.key`, outside
the repository, with owner-only file permissions (`0600`). Future Gemini runs
use it automatically without prompting. Repeat setup to replace it.
Use `--prompt-key` with a cloud action for a temporary override instead.
An existing nonempty exported `GEMINI_API_KEY` takes precedence over the saved
file; `unset GEMINI_API_KEY` removes a stale export from the current terminal.
Local detection and OCR never load the saved key. The configured model is
`gemini-3.5-flash-lite`, with a
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
project. The user reports a successful live Pi Gemini run; controlled quality
and latency checks remain pending. Keep the budget at
**$0 new spending**: verify the intended
project's free tier, billing-disabled state, and model access before a test.
No billing changes, subscription credits, or student credits are assumed.

After pulling the latest code and preparing the approved Pi SDK environment,
save the key once from the repository root:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --save-key
```

Then request a snapshot without entering the key again:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini.yaml --gemini-detect --output artifacts/pi-gemini.png --report artifacts/pi-gemini.json
```

[docs/gemini.md](docs/gemini.md) explains AI Studio key creation, saved keys,
the optional per-run prompt, the approved Pi SDK setup, and snapshot commands.
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
`simulation`, `detection`, `models`, `ocr`, `speech`, `cloud`, `proximity`, and `voice` sections configure their
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
[docs/hardware.md](docs/hardware.md). Pi 5 Gemini OCR and an audible local speech
excerpt passed a user-run check; broader recognition accuracy, offline
Tesseract, and optional neural voice checks remain pending.
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
CLI. `audio/speech_to_text.py` owns explicitly requested short command capture/
transcription; `assistant/command_handler.py` matches a local command allowlist.
`config/`, `examples/`, `scripts/`, `tests/`, and `docs/` hold
configuration, fixtures, diagnostics, verification, and setup notes.

Phase 1 established the foundation. Phase 2 covers camera/detection plus the
explicit one-shot Gemini demonstrations described above. Phase 3 adds requested
Gemini or local OCR and local speech. Gemini reading with audible eSpeak output
has passed on Pi 5; offline Tesseract setup and acceptance remain pending.
Phase 4 adds requested scene snapshots and fresh local two-channel proximity;
the user confirmed functional combined operation. Phase 5 adds M-triggered
voice commands; local Bluetooth recording/playback and voice-triggered
description have user-run evidence, and printed-text output was heard.
Voice-triggered OCR is not established by the supplied log. Measured
hardware accuracy, sustained performance, wearable deployment, and upgrades
remain Phase 6 work at the user's explicit request.
Model class coverage is limited to the selected model's labels; an empty
detection list is not evidence that a path is clear. The full plan, official
references, approval gates, and hardware acceptance record are in
[docs/phase2.md](docs/phase2.md).
