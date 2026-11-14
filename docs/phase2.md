# vidi Phase 2 setup and acceptance

The connected camera is **Camera Module 3 (IMX708)**. The primary path is
Picamera2 capture plus OpenCV DNN CPU inference with MobileNet-SSD. The optional
IMX500 AI Camera implementation is for a different sensor and cannot run on
IMX708. Preserve the partner's existing Pi installation.

## Confirmed hardware and software

The target is a Raspberry Pi 4 running Debian 12, aarch64, Python 3.11.2,
NumPy 1.24.2, PyYAML 6, setuptools 66, Picamera2 0.3.27, and installed OS OpenCV
4.6.0. rpicam reports IMX708. The monitor and keyboard are connected. The
supplied terminal output at commit `827577d` also confirms two camera-only
runs of 3 frames at 640×480 and successful image export. A later run processed
20 live frames with CPU inference ready on all 20 and a final-frame `person`
prediction on a visible hand. In a follow-up controlled test, the user confirmed
that a person box covered the subject's body. Broader object checks and image
orientation/color verification remain pending. The requested Pi 5 with 8 GB RAM
remains unverified.

Use the existing GitHub branch handoff or transfer the source by USB and open
a terminal in the repository root. SSH setup is not required. Read-only checks:

```sh
cat /etc/os-release
uname -m
/usr/bin/python3 --version
/usr/bin/python3 scripts/check_hardware.py --json
rpicam-hello --list-cameras
```

Do not reimage, perform a blanket OS upgrade, change camera configuration, or
replace the working NumPy/OpenCV/Picamera2 stack. The standard-library hardware
diagnostic does not import optional libraries or open devices. Its metadata and
module origins do not prove native ABI compatibility or camera operation.

## Fast first camera check: no pip install

From the repository root, use the confirmed system packages directly:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi.yaml --detector none --max-frames 3 --output artifacts/pi-camera.png
```

This checks Picamera2 capture and an image export without loading any model or
contacting a service. Inspect the exported image for correct orientation and
colors. Stop and report errors before adding packages or changing the OS.

## Approved CPU model and first inference

The author model is [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD),
fixed to commit `bb17b6c3eef36d80be441ae8e5339be66e8e3b7a`. Its download and local
testing were approved. [models.md](models.md) contains the exact fixed URLs,
SHA-256 hashes, MIT license instructions, and explicit download commands.
Transfer verified files or use those approved commands on the Pi. Keep the
upstream license with the assets. Nothing is downloaded at application startup.

Expected files from the repository root:

```text
models/mobilenet-ssd/mobilenet_iter_73000.caffemodel
models/mobilenet-ssd/deploy.prototxt
models/mobilenet-ssd/LICENSE
```

With those local assets, run a real-model load/forward smoke test, then a short
camera run using the same source-tree environment:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --source sample --detector opencv_ssd --model models/mobilenet-ssd/mobilenet_iter_73000.caffemodel --model-config models/mobilenet-ssd/deploy.prototxt --report artifacts/cpu-smoke.json
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi.yaml --max-frames 20 --no-preview --output artifacts/pi-detections.png --report artifacts/pi-cpu.json
```

The first command checks model loading, forward execution, and decoding on a
synthetic frame. It already passed locally on Mac with zero detections; this
is not an accuracy measurement or a Pi result. `config/pi.yaml` selects
`picamera2` plus `opencv_ssd`, requests 640×480, a 2 FPS loop limit, and 20
frames. The supplied Pi report shows 20 live frames processed in 13.26 seconds
(about 1.51 frames/s including the initial frame wait), with about 2.00 FPS
between capture timestamps. Model paths in that YAML start
`../models/` because they resolve relative to `config/`.

The decoder follows the author's 300×300 BGR preprocessing and class ordering.
There are 20 VOC object classes plus background: people, chairs, and some
vehicles are supported; doors and signs are not. Optional label overrides
must preserve all 21 class positions. Other models need matching architecture,
preprocessing, decoder, and labels; changing filenames is insufficient. Empty
detections do not mean a clear path, and confidence is not measured accuracy.

## Optional editable Pi environment

For an installed CLI after the direct check, reuse OS packages:

```sh
/usr/bin/python3 -m venv --system-site-packages .venv-pi
export PYTHONNOUSERSITE=1
.venv-pi/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
.venv-pi/bin/python scripts/check_hardware.py --json
```

The backend minimum is setuptools 64; reported setuptools 66 meets it. If a
local build prerequisite is missing, report the exact error before adding
software. The source-tree command remains available. Never run desktop
`requirements.txt`, `requirements-dev.txt`, or the `desktop` extra on the Pi.
Do not pip-install NumPy, OpenCV, or Picamera2 over the OS stack. No new Python
inference package is needed: OpenCV 4.6.0 supplies the Caffe loader/CPU target.

On macOS, use an ordinary isolated `.venv`, select it in VS Code, and install
`requirements-dev.txt` only with any required package-download approval. Base
metadata declares NumPy/PyYAML; the `desktop` extra adds pip OpenCV and a newer
NumPy floor. The optional `gemini` extra pins the locally tested Google SDK
2.34.0. Package discovery alone does not prove ABI compatibility; record module
origins and verify actual imports/array operations before a camera test.

## Configuration and replay

`config/settings.yaml` is the safe default: one synthetic frame, detector
`none`, no preview/export/network/audio. `config/fixtures.yaml` selects invented
annotations for macOS replay. `config/pi.yaml` is the active IMX708 CPU profile;
`config/pi-imx500.yaml` is only for a separately verified AI Camera.

CLI paths are relative to the terminal directory; YAML paths are relative to
the YAML file. CLI values override YAML. Unknown or invalid settings fail.

| Control | Meaning |
| --- | --- |
| `--source` | `sample`, `image`, `webcam`, `picamera2`, or `imx500` |
| `--detector` | `none`, `fixture`, `opencv_ssd`, or `imx500` |
| `--model`, `--model-config` | Local CPU weights and Caffe prototxt; also `models.object_detection_path` / `object_detection_config_path` |
| `--labels` | Model-aligned optional override, retaining class positions |
| `--threshold`, `--max-detections` | Confidence filter and result limit; defaults 0.55 and 10 |
| `--detections-fixture` | Versioned replay JSON |
| `--max-frames`, `--preview` / `--no-preview` | Bounded processing and optional desktop window |
| `--output`, `--report` | Explicit final-image and JSON exports |
| `--describe`, `--question` | Per-run cloud opt-in and optional typed question |

`examples/detection-fixture.json` has `version: 1`, `image_size: [640, 480]`,
and a `detections` list. Each entry has `class_id`, `label`, `confidence`, and
`bbox_xyxy: [left, top, right, bottom]`. Replay requires matching image size;
reports/previews mark the detections as simulated. Fixture results are not
model predictions. Tests and a local fixture export:

```sh
python -m pytest
python -m vidi --config config/fixtures.yaml --max-frames 3 --output artifacts/fixture.png --report artifacts/fixture.json
```

## Offline bench acceptance

After the short camera and model checks, use controlled scenes with consenting
participants and objects supported by the labels. Run a later 100-frame test
with networking unavailable or disabled by the operator's normal controls:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi.yaml --max-frames 100 --no-preview --report artifacts/pi-offline-100.json
```

Record OS/package/model versions, hashes, import origins, requested/actual frame
size, startup/capture duration, successful frames, observed rate, thresholds,
and report paths in [hardware.md](hardware.md). Check image colors and box
alignment; missing model/config files, camera loss, timeout, headless use,
and Ctrl+C must give useful errors and release resources. Verify the absence
of a key, SDK, or network leaves local detection results available.

The supplied Pi logs confirm camera-only capture/export and a 20-frame live
CPU inference run, with all 20 frames ready and a final-frame `person` prediction
(score 0.654). Camera-worker startup was 3.386 seconds; processing was 13.260
seconds. The report's 2.00015 FPS rates use capture timestamp intervals, whereas
20 / 13.260 gives about 1.51 frames/s over the full processing loop. Neither
measures isolated inference latency or maximum detector speed; the run requests
a 2 FPS limit. The initial prediction covered a visible hand; the user later
confirmed a box covering a person's body in a separate controlled person test.
No follow-up JSON metrics were supplied. This confirms one person example;
broader recognition accuracy and longer offline acceptance remain pending.
The current software suite has 351 passing tests and one sandboxed
shared-memory skip; that test passed separately outside the sandbox, for 352
validated tests. Dependency consistency passed. Mocked hardware/SDK tests and
the actual Mac model-forward smoke test do not replace on-device acceptance.

## Optional IMX500 profile

Only a confirmed IMX500 sensor can use `config/pi-imx500.yaml` and its local
`/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`.
The decoder accepts that baseline filename, with a 320×320 model contract;
other networks require their own decoder. Labels come from an explicit file
or embedded metadata. Missing usable labels fail clearly. Preserve blank/dash
entries and class IDs rather than shifting an arbitrary COCO label mapping.
Do not install IMX500 packages or download its model for the connected IMX708.

References: [AI Camera documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html),
[model catalogue](https://github.com/raspberrypi/imx500-models), and the official
[object detection example](https://github.com/raspberrypi/picamera2-examples/blob/main/examples/imx500/imx500_object_detection_demo.py).

## Optional Gemini demonstration and approval boundaries

This section records the original Phase 2 approval state. User-triggered
Gemini test snapshots and the Pi SDK download were approved during the later
Gemini integration; use [gemini.md](gemini.md) for the current setup and scope.

The approved Google SDK 2.34.0 was installed/tested locally on Mac using mocked
HTTP and blocked sockets. No live Gemini API request or image upload has been
approved or performed. API-project setup, current tier/terms review, and live
image consent remain separate approval steps. The budget is **$0 new spending**:
verify a free-tier project with billing disabled and chosen-model eligibility;
otherwise skip cloud use. Do not assume Google AI Pro/student benefits provide
API credits. Verify any credit offer separately.

`--describe` explicitly opts one run into one request after camera resources
close. It sends the final unannotated frame, resized to a maximum 1024-pixel
edge, and an optional typed question. `--question` requires `--describe`.
Configuration alone never initiates a request. The model is
`gemini-3.5-flash-lite`, timeout 15 seconds, and at most 256 output tokens.
Set `GEMINI_API_KEY` securely; `.env` is not autoloaded. Never commit or log a
key. Use a permitted, non-sensitive demonstration image; avoid personal
papers and bystanders. Cloud text is separate from local predictions and is
not spoken. Failures preserve local results. Do not enable continuous uploads.

Verify the current [model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
[image guide](https://ai.google.dev/gemini-api/docs/image-understanding),
[billing](https://ai.google.dev/gemini-api/docs/billing), and
[data-use terms](https://ai.google.dev/gemini-api/terms) before a live request.
Check demonstration criteria against the official
[HackUMass prizes](https://www.hackumass.com/#prizes); no award or credit is assumed.
OCR, speech, voice commands, wearable integration, and later performance claims
remain outside this implementation. The prototype supplements established
mobility aids and cannot provide precise distance or guaranteed navigation safety.
