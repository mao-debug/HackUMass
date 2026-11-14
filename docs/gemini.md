# vidi Gemini setup and live identification

vidi's snapshot cloud actions are `--gemini-detect`, which produces object labels
and boxes; `--describe`, which produces a short scene description; and
`--gemini-read`, which transcribes printed text for Phase 3. Each sends
one final unannotated image only after the camera closes. Increasing
`--max-frames` does not create additional uploads. No local detection weights,
Tesseract, microphone, or Gemini speech-generation API is needed.

The separate `--continuous --gemini-live` action keeps the camera open and
explicitly authorizes repeated selected images to Google. This repeated-frame
use has been approved with a five-second minimum interval, one request at a
time, stop on error/quit, a verified free-tier project, and no billing changes.
No new package or local-model download is needed.

For printed text, use the existing SDK and saved key:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini-ocr.yaml --gemini-read --report artifacts/pi-gemini-reading.json
```

Stop any live camera run first. `--gemini-read` sends one raw snapshot after
camera cleanup, returns text with model/timing metadata, and can be followed
by explicit local `--speak` or `--speech-output FILE.wav`. It requires detector
`none`, cannot be combined with other reading/cloud/live actions, and never
silently falls back to Tesseract. The 60-second OCR timeout and 2048 output
tokens are independent of the description/detection budgets. Full text is
checkpointed before speech; missing audio hardware or engines cannot discard
it. See [Phase 3 setup](phase3.md) for Bluetooth routing, approved eSpeak NG
installation, offline reading, and bench checks. Gemini voice-command input
is a separate Phase 5 task; this action does not record a microphone.

The detection path is implemented and checked with offline tests. Approval covers
the pinned Pi SDK and required dependencies from PyPI in `.venv-pi`, plus
user-triggered non-sensitive test snapshots through a verified free-tier
project, with no billing changes. The Mac SDK is already installed and its
existing checks used mocked transport. The user reports a successful live Pi
Gemini snapshot run; controlled live-mode recognition-quality and latency checks
remain pending.

## Responsive live preview with repeated identification

After pulling the update, use your existing `.venv-pi` and saved key:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-live.yaml --continuous --gemini-live --preview --report artifacts/pi-live-gemini.json
```

This profile requests a 640×480 camera feed at 30 FPS. The supplied live Pi run
measured 11.21 FPS; the requested rate is not guaranteed. That run's only Gemini
request timed out, so recognition was unavailable rather than successfully empty.
The old `config/pi.yaml` deliberately requests
2 FPS and runs the temporary 20-class local detector. Changing to live Gemini
addresses its limited vocabulary; it does not establish prediction accuracy.

The left pane stays on the current raw camera image. The right pane shows the
selected image and explicit pending, unavailable, or successful-result state.
Successful results show labels, boxes, frame ID, and current age. "No objects
identified" is used only for a successful empty response. No old boxes are
placed on a newer live image. Capture and preview do not
wait for inference. One pending image can be replaced by a newer one; older
frames never build up in a queue.

Gemini waits `cloud.live_interval_s` (default and minimum five seconds) after
one detection completes before starting another. There are no overlapping
requests or automatic retries. Any unavailable/failed result stops further
uploads; the live camera remains active and shows the failure state. Fix the
cause and restart the command to resume. No automatic fallback to the old
local detector occurs. Live requests now use a **60-second request timeout**
instead of 15 seconds; snapshot profiles keep 15 seconds. This gives a slow
response more time without blocking the camera, but does not guarantee success.
The saved key and YAML never opt in by themselves:
`--gemini-live` and `--continuous` must both be present.

Press q/Esc or Ctrl+C to stop new submissions. An in-flight request may finish;
the wait for the worker is bounded and no queued frame is sent afterward.
This does not record a video file or generate continuous speech. If `--output`
is supplied, it saves the last analyzed image with boxes associated with that
same frame. Live reports use schema version 2 and distinguish
`last_preview_frame_id`, `last_result_frame_id`, `last_analyzed_frame_id`,
and `output_frame_id`. An unavailable request does not count as a successful
analysis; `cloud_timeout_s` records the timeout used for the run.
`frames` retains the latest 120 received analysis results, while capture and
analysis totals span the whole run. The schema-1 snapshot reports below are
unchanged.

To compare camera speed without inference or uploads:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-live.yaml --continuous --detector none --preview
```

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
   selected project. Copy it privately for vidi's hidden prompt below.

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

## Recommended: save the key once on this device

On the Pi, run this setup command from the repository root:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --save-key
```

On Mac, use `.venv/bin/python -m vidi --save-key` instead. Paste only the raw
key at the hidden prompt and press Enter. Setup saves the key and exits without
opening the camera, loading the Google SDK, or contacting Google. The flag takes
no argument and must be used by itself.

The key is stored in **plain text, not encrypted**, at
`~/.config/vidi/gemini.key`, outside the repository. The vidi directory is
owner-only (`0700`) and the file is readable and writable only by your user
(`0600`). This keeps it across terminal sessions and reboots without placing it
in Git, YAML, shell history, images, or reports. Do not share or copy that file
into the repository. Run the same setup command again to replace the saved key;
an interrupted or failed save preserves the previous key.

After setup, omit `--prompt-key` from Gemini commands. vidi loads the saved key
only for an explicitly requested `--gemini-detect`, `--gemini-read`, `--describe`, or
`--continuous --gemini-live` action.
Saving a key does not authorize automatic uploads or continuous requests.

Credential priority is `--prompt-key` for a temporary override, then an existing
nonempty exported `GEMINI_API_KEY`, then the saved file. If an old exported key
is overriding the saved one, run `unset GEMINI_API_KEY` in that terminal.
`.env` files are not loaded. To stop using the saved key, delete
`~/.config/vidi/gemini.key`; this does not revoke it at Google.

## Optional: enter a key for one run only

Use `--prompt-key` with a Gemini snapshot or live action. The flag takes **no
key argument**. Run the command in an interactive terminal, such as the Pi's
Terminal application or VS Code's terminal, rather than a non-interactive
output panel or pipe.

When vidi displays its key prompt, paste **only the raw API key**, without an
assignment such as `GEMINI_API_KEY=`, quotation marks, or another command.
Press Enter. No characters are shown while the key is entered; this is
expected. The key is read during that vidi run and applies only to its process.
vidi restores any previous environment value afterward and does not write
the prompted key to disk or shell history.

If the terminal cannot provide hidden input, vidi refuses the prompt rather
than falling back to visible entry. `--prompt-key` is accepted only for the
explicit Gemini actions; it is not used for local OCR or offline detection.
Never paste the key into chat, screenshots, GitHub, YAML, logs, or reports.

Without `--prompt-key`, vidi uses the exported or saved key described above.
Neither setup method needs a shell variable assignment.

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

On the Pi, after the SDK is available, the latest code has been pulled, and
the key has been saved once, run this command from the repository root:

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
`--gemini-detect`, `--gemini-read`, `--describe`, or `--read-text` for a run.

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
validation reports **687 passed, 4 skipped**, outside the restricted sandbox,
including mocked private-prompt and environment-restoration checks.
The skips are optional Tesseract/eSpeak NG engine checks; Gemini's SDK tests
run with blocked sockets, fake credentials, and local MockTransport responses,
including one-request behavior for 200/429/503 outcomes. Dependency consistency
and whitespace checks passed. These tests do not establish live Gemini
recognition quality, paid account setup, or audible speech.
