# vidi one-shot Gemini setup

vidi has two explicit cloud actions: `--gemini-detect` produces object labels
and boxes, while `--describe` produces a short scene description. Each sends
one final unannotated image only after the camera closes. Increasing
`--max-frames` does not create additional uploads. No local detection weights,
Tesseract, microphone, or Gemini speech-generation API is needed.

The detection path is implemented and checked with offline tests. Approval covers
the pinned Pi SDK and required dependencies from PyPI in `.venv-pi`, plus
user-triggered non-sensitive test snapshots through a verified free-tier
project, with no billing changes. The Mac SDK is already installed and its
existing checks used mocked transport. Pi SDK installation, local key setup,
and a live image request have not yet been performed; the commands below are
the approved setup and test path.

## Prepare an API key without sending an image

Follow Google's current [API-key guide](https://ai.google.dev/gemini-api/docs/api-key)
and open the [AI Studio API Keys page](https://aistudio.google.com/apikey).

1. Sign in with the intended Google account and review the presented terms.
2. Select the intended Google Cloud project. AI Studio can create a default
   project/key for new users after terms acceptance. Existing Cloud users may
   need **Dashboard → Projects → Import projects**, then return to API Keys.
3. Inspect that project's tier and billing status. For this $0-spend demo,
   use a verified free-tier project with no billing account attached. Do not
   click an upgrade, enable billing, add payment details, or prepay credits.
4. If a usable key is not already present, use **Create API key** in the
   selected project. Copy it privately for the terminal step below.

Google's current guide says new AI Studio keys use authorization keys; old
unrestricted standard keys are rejected. Use the current key-creation flow
rather than weakening restrictions on an old key. Some imported projects need
verification before free-tier access. These details come from the
[official key guide](https://ai.google.dev/gemini-api/docs/api-key).

Confirm the selected project remains eligible for the configured
`gemini-3.5-flash-lite` model and its free quota using the current
[model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
[billing guide](https://ai.google.dev/gemini-api/docs/billing), and AI Studio
project/rate-limit views. Model availability is account-dependent. If access
requires billing or verification that cannot be completed within the approved
scope, keep the offline demo. Do not assume Google AI Pro or student offers
provide API credits.

## Set the key privately in Bash

Use the same terminal shell that will launch vidi. These steps store the key
only in that shell's environment. They do not make a Gemini request. Start
Bash first, including on a Mac whose usual shell is zsh:

```sh
bash
```

Run this command by itself:

```sh
read -r -s GEMINI_API_KEY
```

Paste the raw key without quotation marks and press Enter. Input is hidden.
**Only after entering the key**, run the separate export command:

```sh
export GEMINI_API_KEY
```

Do not paste all three steps at once: a following command could be consumed as
the key input. Do not echo the variable, include the key in shell commands,
paste it into chat, or put it in screenshots, GitHub, YAML, logs, or reports.
Use ordinary ASCII command text; copied curly quotation marks caused earlier
shell trouble. The commands above need no quoted key assignment.

vidi reads `GEMINI_API_KEY`; it does not load `.env` files. A key exported in
one terminal is not automatically available in another terminal or a separately
launched VS Code process. Remove the variable when finished:

```sh
unset GEMINI_API_KEY
```

If a key is exposed, revoke it through AI Studio and create a replacement;
do not include the exposed value when reporting the problem.

## Approved Pi SDK environment

Keep the working Pi OS NumPy, OpenCV, Picamera2, and Python installation.
Never install desktop requirements, pip NumPy/OpenCV, or pip Picamera2 over it.
Use the ignored `.venv-pi` environment with access to system camera packages.
If it does not already exist, create it from the repository root:

```sh
/usr/bin/python3 -m venv --system-site-packages .venv-pi
```

Install the approved pinned SDK and its required dependencies from PyPI in
that environment:

```sh
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m pip --isolated install --index-url https://pypi.org/simple google-genai==2.34.0
```

If package resolution proposes replacing the camera/NumPy/OpenCV stack, stop
and inspect the change rather than substituting a desktop environment. Use
`PYTHONPATH=src` for the source-tree commands below; no second editable project
installation is required. The Mac's existing `.venv` already has the approved
SDK and needs no new SDK download for this work.

Confirm the installed version locally without an API request:

```sh
.venv-pi/bin/python -m pip show google-genai
```

## One approved object-detection request

The commands in this section **send the chosen test image to Google**. These
user-triggered test snapshots are approved within the verified free-tier,
no-billing-change scope. Confirm that the intended project and model have
free quota, and choose a non-sensitive test scene. Free-service data use can
include product improvement and human review;
check the current [Google terms](https://ai.google.dev/gemini-api/terms), and
exclude personal papers and bystanders from the demonstration scene.

On the Pi, after the SDK and key setup above, run from the repository root in
the Bash session holding the exported key:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini.yaml --gemini-detect --output artifacts/pi-gemini.png --report artifacts/pi-gemini.json
```

On Mac, use a chosen non-sensitive local test image and the existing desktop
`.venv`:

```sh
.venv/bin/python -m vidi --image path/to/test-scene.png --gemini-detect --output artifacts/mac-gemini.png --report artifacts/mac-gemini.json
```

`--gemini-detect` selects the `gemini` detector automatically. Merely setting
`models.object_detection_backend: gemini`, or passing `--detector gemini`,
does not authorize a request: the explicit flag is required. Choose one of
`--gemini-detect`, `--describe`, or `--read-text` for a run.

The Pi cloud profile requests a 1280×960 camera image at 5 FPS, one final frame,
a one-second settling interval, and a supported single autofocus cycle with
a four-second wait. Startup/capture deadlines remain 30/five seconds. Camera
focus is preparation, not an object-recognition guarantee. Camera resources
are released before the network request begins.

The configured cloud model is `gemini-3.5-flash-lite`. The upload is resized to
at most 1024 pixels on its longest edge, with a 15-second request timeout and
`cloud.detection_max_output_tokens: 1024`. The default result limit is 10
objects. Inspect the final image/report after completion; this mode is not a
live per-frame cloud overlay.

With `--preview`, q/Esc finishes capture and proceeds with the requested action
on that final frame; Ctrl+C cancels. A returned Gemini overlay is displayed for
one second. Use `--output` to retain it for longer inspection.

## Interpret the detection report

The response supplies generated labels and integer normalized box coordinates
on a 0–1000 scale. The application validates them and converts accepted boxes
to the original image's pixel coordinates. Reports use `bbox_xyxy` for these
pixel boxes. `class_id` and `confidence` are both `null`: they are not SSD class
indices or model confidence scores. Explicit `--threshold`, `--model`, and
`--model-config` options are rejected for Gemini instead of being ignored.

The report's `last_inference` object records the final cloud result, including
its model, elapsed time, and any error. Only the last captured frame receives
that cloud result; earlier frame records do not imply additional requests.

Generated labels are not limited to MobileNet-SSD's VOC class list. This does
not promise that every object is found or correctly named. Boxes may be
inaccurate and objects may be omitted. Empty results cannot establish that a
path is clear or safe.

There are no automatic retries and no automatic fallback to a local detector.
An API, network, quota, or response-validation failure records the cloud result
as `unavailable`; when requested, the local image and JSON report are retained.
The offline `config/pi.yaml` CPU pipeline is a separate explicit option:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi.yaml --output artifacts/pi-offline.png --report artifacts/pi-offline.json
```

That offline command needs the already approved local model assets, but no
key, Google SDK, or network connection. No cloud failure silently substitutes
its output for the requested Gemini result.

## Descriptions and optional local speech

`--describe` is an alternative one-image request. It accepts a typed
`--question`, which is not voice input, and uses the description output limit
of 256 tokens. It shares the model, image-edge limit, timeout, and approved
free-tier test scope above. For a Pi description, override the profile's object
detector with `none`:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini.yaml --detector none --describe --report artifacts/pi-description.json
```

After the local speech dependency and output device are separately ready, add
either `--speak` or `--speech-output artifacts/gemini.wav` to a cloud action.
Both are explicit and mutually exclusive. The returned detection summary or
description is passed to local TTS; no Gemini TTS endpoint is called. Existing
speech limits remain 150 words per minute, a 60-word excerpt, and a 45-second
process timeout. WAV export uses local eSpeak NG. An engine-completed status
does not verify audibility. Both cloud actions check interruption before
starting local speech; Ctrl+C can cancel the command. Tesseract is not needed for these cloud actions;
Pi eSpeak NG installation and audio acceptance remain pending as described in
[phase3.md](phase3.md).

Offline CPU/camera evidence remains in [hardware.md](hardware.md). Current Mac
validation reports **632 passed, 4 skipped**, outside the restricted sandbox.
The skips are optional Tesseract/eSpeak NG engine checks; Gemini's SDK tests
run with blocked sockets, fake credentials, and local MockTransport responses,
including one-request behavior for 200/429/503 outcomes. Dependency consistency
and whitespace checks passed. These tests do not establish live Gemini
recognition quality, paid account setup, or audible speech.
