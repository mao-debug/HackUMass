# vidi Phase 2 plan and acceptance

This document records the selected Phase 2 implementation and the remaining
on-device acceptance work. Software fixtures and mocked tests are not evidence
that the physical camera, model, or cloud service has passed a live test.

## Selected scope

- Run real local detection on the available Raspberry Pi 4 and AI Camera
  (IMX500). The partner's monitor and keyboard are connected; inspect and
  preserve the current OS before changing it. The requested Pi 5 remains
  unverified.
- Use the official SSD MobileNet FPNLite 320 model as the baseline. Load its
  local packaged `.rpk` network, parse output metadata, and expose filtered
  labels, confidences, and image-space boxes. Do not train, convert, or download
  a model automatically.
- Develop on macOS using synthetic/image frames and clearly identified fixture
  replay. Fixture boxes are invented development data, not detections from a
  Mac inference model.
- Offer an explicit one-shot Gemini text description for the demonstration.
  Keep it separate from local predictions. No audio, OCR, or voice commands are
  added in this phase.
- Keep local detection usable without a network connection. Default execution
  stays hardware-free: one synthetic frame, detector `none`, no preview,
  export, or cloud request.

No precise distances, guaranteed navigation safety, fixed accuracy, or achieved
FPS are claimed. Detection supports only the chosen model's label set; an empty
result does not establish that a scene is safe.

## Configuration and data contracts

`config/settings.yaml` supplies defaults; `config/fixtures.yaml` selects replay;
`config/pi.yaml` selects the AI Camera. CLI values override YAML. CLI paths are
relative to the current working directory; YAML paths are relative to that
file. Unknown or invalid settings fail rather than being silently ignored.

| Setting | Default or meaning |
| --- | --- |
| `camera.source` | `sample`; alternatives `image`, `webcam`, `imx500` |
| `camera.width`, `height`, `fps` | `640`, `480`, `5.0`; requested capture/loop settings, not measured performance |
| `camera.startup_timeout_s`, `capture_timeout_s` | `300.0`, `5.0`; startup accommodates initial model loading |
| `simulation.max_frames`, `preview` | `1`, `false`; Pi bench config requests 100 frames |
| `simulation.output_path`, `report_path` | `null`; export only when requested |
| `models.object_detection_backend` | `null`/`none`, `fixture`, or `imx500` |
| `models.object_detection_path` | Local supported baseline `.rpk` path, required for selected IMX500 detection |
| `detection.confidence_threshold`, `max_detections` | `0.55`, `10` |
| `detection.labels_path`, `fixture_path` | `null`; optional model-aligned labels and explicit fixture input |
| `models.scene_description_backend` | `null` or `gemini`; this never opts a run into uploading |
| `cloud.model` | `gemini-3.5-flash-lite`; verify current availability/free-tier eligibility before a call |
| `cloud.timeout_s`, `max_output_tokens`, `max_image_edge` | `15.0`, `256`, `1024` |

The CLI adds `--source imx500`, `--detector {none,imx500,fixture}`, `--model`,
`--labels`, `--threshold`, `--max-detections`, `--detections-fixture`, and
`--report`, while preserving frame count, preview, and image output controls.
`--describe` opts in once for the run; `--question` requires that flag.

Fixture JSON has the following versioned shape, illustrated by
`examples/detection-fixture.json`:

```json
{
  "version": 1,
  "image_size": [640, 480],
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.95,
      "bbox_xyxy": [64, 160, 256, 384]
    }
  ]
}
```

Coordinates describe the declared fixture image size; frame dimensions must
match that size for replay. Reports and previews must
identify fixture results as simulated. Real detection boxes must correspond to
the returned camera frame rather than being treated as distance measurements.

## AI Camera model and labels

The selected default model path is:

```text
/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
```

Use compatible OS-provided Picamera2/IMX500 components and a verified local model
package. The decoder currently accepts only the baseline filename shown above;
`--model` can select its location but does not make arbitrary networks
compatible. Other models require an explicitly implemented decoder. Source and
detector selections must agree; an incompatible input,
missing model, or unsupported output layout must fail clearly. No hidden model
download or substitute detector is allowed.

Labels come from an explicit `detection.labels_path`/`--labels` override or the
network's embedded labels. If neither is usable, stop with a clear error.
Preserve class IDs and blank/dash entries in supplied label lists; do not guess
an arbitrary COCO offset or renumber categories. If the official example label
file is needed, obtaining it is a separate approved download, and its alignment
with this network must be checked.

The official [AI Camera documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html),
[model catalogue](https://github.com/raspberrypi/imx500-models),
[object detection example](https://github.com/raspberrypi/picamera2-examples/blob/main/examples/imx500/imx500_object_detection_demo.py),
and [example COCO labels](https://github.com/raspberrypi/picamera2-examples/blob/main/examples/imx500/assets/coco_labels.txt)
are the compatibility references. Consult the versions matching the installed
camera stack; the example labels are not automatically bundled or fetched.

## Inspect the existing Pi before changes

Use the connected desktop to inspect the current installation. These commands
read local state and do not install software or open a camera:

```sh
cat /etc/os-release
uname -m
/usr/bin/python3 --version
/usr/bin/python3 scripts/check_hardware.py --json
dpkg-query -W python3-numpy python3-opencv python3-yaml python3-picamera2 python3-setuptools python3-venv imx500-all
```

Missing package entries are useful findings, not permission to install them.
Record OS release, architecture, available storage, board model/RAM, Python
executable/version, existing package versions, and model/label files. Check
Python is at least 3.10 and compare the installed OS/camera stack against the
official compatibility instructions. Inventory the cable and connection before
any camera run.

Preserve the partner's installed OS and files. Do not reimage, perform a blanket
upgrade, or change boot configuration during inspection. If compatibility
cannot be established, stop and propose a supported image on a separately
approved spare microSD card. Package installs, model/label downloads, and
hardware changes require their own concrete approval.

## Dependency policy and approved-install procedure

macOS uses an ordinary isolated `.venv` and the `desktop` extra. Base metadata
declares `numpy>=1.24,<3` and `PyYAML>=6,<7`; `desktop` adds `numpy>=1.26,<3`
and `opencv-python>=4.10,<5`. Pytest is a development requirement. The optional
`gemini` extra pins the locally tested `google-genai==2.34.0`. The SDK was
installed in the Mac environment with approval; no Pi packages, camera model,
or labels have been downloaded or installed here.

On the Pi, use one compatible OS package stack for `python3-numpy`,
`python3-opencv`, `python3-yaml`, `python3-picamera2`, `python3-setuptools`,
`python3-venv`, and `imx500-all`. The IMX500 package can bring model assets and
other runtime dependencies. Review the proposed package changes and downloads
on the actual OS, then obtain approval before running an installation command.
Do not overlay pip NumPy, OpenCV, or Picamera2 on this stack. apt's `cv2` import
does not need to advertise a PyPI distribution named `opencv-python`.

Once the compatible prerequisites are present, create a separate environment
using the OS interpreter:

```sh
/usr/bin/python3 -m venv --system-site-packages .venv-pi
export PYTHONNOUSERSITE=1
.venv-pi/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
.venv-pi/bin/python scripts/check_hardware.py --json
```

This editable-install command cannot fetch dependencies or an isolated build
backend. Verify a compatible local setuptools backend (minimum 64) first; stop
on a missing backend rather than removing these safeguards. Never run
`requirements.txt`, `requirements-dev.txt`, or the `desktop` extra on the Pi.
`PYTHONNOUSERSITE=1` prevents user-site packages from shadowing the intended
system stack.

The diagnostic reads metadata without importing optional libraries; it cannot
prove ABI compatibility. Before opening the camera, run this explicit local
import/array smoke test and record the printed versions and origins:

```sh
PYTHONNOUSERSITE=1 .venv-pi/bin/python -c 'import numpy as np, cv2, yaml, picamera2; print("numpy", np.__version__, np.__file__); print("cv2", cv2.__version__, cv2.__file__); print("yaml", yaml.__version__, yaml.__file__); print("picamera2", picamera2.__file__); print(cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY).shape)'
```

NumPy, OpenCV, and the camera libraries should resolve to the intended OS
`dist-packages`, with compatible versions. If imports fail or resolve to an
unexpected pip/user environment, stop and repair the environment deliberately.
Do not fix an ABI failure by upgrading NumPy alone.

## Local bench acceptance

Run software tests and fixture replay on macOS first:

```sh
python -m pytest
python -m vidi --config config/fixtures.yaml --max-frames 3 --output artifacts/fixture.png --report artifacts/fixture.json
```

On the Pi, after the prerequisite checks, run with networking unavailable or
disabled through the operator's normal controls:

```sh
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m vidi --config config/pi.yaml --max-frames 100 --no-preview --report artifacts/pi-offline-100.json
```

Use a controlled bench scene with consenting participants and examples of
objects supported by the labels. Record the requested and actual frame sizes,
number of successful frames, startup time, elapsed processing time, observed
rate, confidence settings, detection counts, OS/package/model versions, and
model/labels provenance. Record timing distributions only when measured; do
not infer accuracy from confidence values or set an unmeasured FPS guarantee.

| Check | Acceptance observation to record |
| --- | --- |
| Default sample | One frame, detector disabled, no camera/network/audio |
| Fixture replay | Simulated marker, expected boxes/filtering, local image and JSON exports |
| Real offline inference | 100 camera frames with model-aligned labels and image-aligned boxes; no cloud dependency |
| Frame integrity | Pixel colors and box alignment checked visually against the controlled scene |
| Config/model failures | Missing model/labels, wrong backend/source, invalid fixtures and thresholds fail clearly |
| Camera loss/timeouts | Useful failure, bounded configured waits where supported, resources released |
| Exit and headless use | Ctrl+C and normal exit release capture; `--no-preview` works without a desktop window |
| Cloud absent/failing | Local predictions remain available with no key, no SDK, quota error, or timeout |

This list is a test procedure, not a completed result. Update
[hardware.md](hardware.md) with actual observations and the report paths. Pi 5
compatibility and wearable use require later physical checks.

## Optional one-shot Gemini demonstration

The selected SDK path uses Google's current Gemini image-understanding API.
The configured model is `gemini-3.5-flash-lite`; confirm access and the applicable
free tier before relying on it. Keep the optional SDK lazy so local detection
does not require it. An SDK install, API-project setup, and the first live
upload are distinct approval steps; none is authorized by running local tests.
On the Pi, review SDK dependencies separately and preserve the OS camera stack.

SDK 2.34.0 normalizes a zero retry setting during client initialization. The
client restores the public `HttpOptions.retry_options.attempts` setting before
constructing the lazy Interactions interface. Offline tests with the installed
SDK verify exactly one request for successful, rate-limited, and server-error
responses. Keep the tested SDK pin until these compatibility tests pass with a
deliberately selected replacement. No private SDK attributes are modified.

The budget is **$0 new spending**. Use a verified free-tier API project with
billing disabled; do not enable paid billing, purchase credits, or assume a
Google AI Pro/student subscription includes API credits. Verify any student
offer separately. If the selected model is unavailable without spending, skip
the cloud demonstration and retain offline local detection.

After approval, SDK setup, and a non-sensitive test image are ready, set
`GEMINI_API_KEY` through a secure shell environment. `.env` files are not loaded
automatically. Do not print the key or include it in commands committed to the
repository. An example invocation, which makes a live upload when properly
configured, is:

```sh
python -m vidi --image artifacts/demo-input.png --detector none --describe --question "Briefly describe the scene." --report artifacts/demo-description.json
```

`--describe` is an explicit per-run opt-in. It sends the last unannotated frame
after the capture context closes, with a maximum image edge of 1024 pixels,
15-second configured request timeout, and at most 256 output tokens. Do not
send annotated detection overlays as visual evidence. The response is text,
identified as a cloud description separately from local model predictions;
it does not trigger speech. Never turn this into continuous uploading.

Use an image without personal documents, bystanders, or other sensitive content,
and obtain permission for the demonstration image. Review current provider
data-use terms and quotas with the project owner before the first request;
free and paid tiers can have different treatment. Avoid logging prompts,
images, credentials, or raw provider responses unnecessarily. Test cloud
failure with mocks before any approved live call.

The official [model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
[image-understanding guide](https://ai.google.dev/gemini-api/docs/image-understanding),
[API overview](https://ai.google.dev/gemini-api/docs/interactions-overview),
[billing documentation](https://ai.google.dev/gemini-api/docs/billing), and
[API terms](https://ai.google.dev/gemini-api/terms) are the references to verify
at setup time. For the demonstration, confirm the current criteria against the
official [HackUMass prize list](https://www.hackumass.com/#prizes); no prize,
eligibility, API credit, or award is assumed by the implementation.

## Remaining approval and evidence gates

Local software validation on November 14, 2026 used macOS arm64 and Python
3.14.6: 259 tests passed and one real shared-memory test was skipped by the OS
sandbox. That skipped test passed separately outside the restriction with a
spawned fake worker, no camera access, and no network. Google SDK 2.34.0 passed
offline request/response and single-attempt error tests with blocked sockets.
Fixture image/JSON export, the installed CLI, dependency consistency, and
whitespace checks also passed. No physical camera or live cloud result is
represented by these software checks.

The approved SDK download and local compatibility tests are complete. System
package installation on the Pi, model/label transfer, paid service activation,
and live image upload have not been performed. Obtain approval for each
concrete proposed external action. After hardware acceptance, keep OCR, audio/voice,
wearable integration, and performance claims in their later approved phases.
