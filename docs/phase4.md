# vidi Phase 4: requested scenes and local proximity

Phase 4 combines a requested camera scene description with **separate left and
right ultrasonic channel feedback**. The Pi 5 camera remains responsive while
cloud, USB, and optional local speech work runs separately. The initial
interface is a keyboard: descriptions and audio are on request. Microphone
commands remain Phase 5; wearable deployment, portable runtime, and sustained
power/cooling optimization remain Phase 6.

The existing Pi 5 capture, Gemini live detection, and Gemini OCR with audible
eSpeak output have supplied acceptance evidence in [hardware.md](hardware.md).
Those results do not validate the new ultrasonic workflow. Photos show an
ESP32 board, USB connection, and pin labels, but do not establish resistor
values, echo voltage, acoustic accuracy, or power quality. **Physical Phase 4
sensor, scene/audio, and performance acceptance are still pending.**

## Prepare the existing installation

First fetch and select the published Phase 4 branch on the Pi:

```sh
cd ~/vidi
git fetch origin
git switch feat/vidi-phase4
git pull --ff-only
```

Run these commands one at a time and resolve any Git error before continuing.
Work from the repository root with the retained `.venv-pi`. Preserve its
OS-provided camera/OpenCV/NumPy stack and existing Google SDK/saved key.
`pyserial==3.5` is optional and only required for the physical USB reader.
Its download is approved and installed on Mac; this Pi command is approved:

```sh
cd ~/vidi
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m pip install --no-deps pyserial==3.5
PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/check_sensors.py --list
```

Listing discovers port metadata without opening a device, taking a camera
image, playing audio, or contacting Gemini. Use the actual port from that
output; `/dev/ttyUSB0` in the commands below is only an example. A stable
`/dev/serial/by-id/` path can be used when available. No automatic port
selection or scan-and-open behavior is used. Close Arduino Serial Monitor
before another program uses the same port. Report access errors before changing
system permissions or adding packages outside the existing approvals.

## Upgrade the ESP32 protocol first

Use [esp32code/esp32code.ino](../esp32code/esp32code.ino), adapted from the
partner's sketch. The pin assignments remain left TRIG/ECHO **25/34** and
right **26/35**. Keep one verified echo divider per 5 V HC-SR04 and a common
ground. Follow the [wiring and build instructions](../esp32code/README.md).
The actual target cross-compile passed with **Arduino CLI 1.6.0** and
**Arduino-ESP32 3.3.12**, generic **ESP32 Dev Module**: DIO flash at 40 MHz,
4 MB flash, PSRAM disabled, upload speed 115200, and all warnings enabled.
The clean rebuild completed with no warnings and used **276476 program bytes
(21%)** and **22788 static RAM bytes (6%)**. This is a compiler
result, not a board upload, wiring test, or physical sensor pass. No GPIO or
serial device was opened.

The approved tools/config remain under ignored `artifacts/tools/arduino/`;
intermediates are in `artifacts/firmware-build/` and local-only binaries in
`artifacts/firmware/`, including `esp32code.ino.bin`. No generated firmware
binary is committed. The firmware README records the exact FQBN and command.

The old streaming firmware is incompatible: it emits fused coordinates,
TTC/confidence, and a `clear` hazard when readings are absent. The host rejects
that legacy stream instead of treating it as a working protocol-v1 sensor.

| Protocol rule | Version 1 |
| --- | --- |
| USB serial | 115200 baud |
| Identification | `H1 ` followed by exactly 16 lowercase hexadecimal characters and newline |
| Fresh sample request | `R1 ` with a new token; samples begin after the command |
| Hello reply | `v: 1`, `type: hello`, matching `id`, `fw: vidi_ultrasonic`, random eight-hex `boot`, channels `left`/`right` |
| Sample reply | Matching token/boot, uint32 `seq`, and independent `left`/`right` results |
| Each channel | `status`, integer `distance_mm` or null, and integer `age_ms` |
| Channel statuses | `ok`, `no_echo`, `out_of_range`, `error` |
| Accepted measurements | Rounded integer 20–3000 mm only for `ok`; other statuses have null distance |
| Bounds | 64-byte commands, 512-byte JSON replies including newline; malformed/oversized input rejected |
| Busy behavior | Matching `type: error`, `code: busy`; no queued or cached replacement request |

The firmware retains interrupt acquisition and freezes the first complete
pulse under a shared main/ISR critical section. Its echo deadline is 30 ms;
global trigger spacing is at least 100 ms, for about five sequential L/R pairs
per second. The fixed conversion assumes 20 C. There is no smoothing history
that converts a missed echo into a new observation, no center sensor, and no
triangulation, predicted range, TTC, confidence score, or clearance output.
Sequential firing reduces interference but does not prove its absence.

After confirming wiring and the firmware upload, request a bounded local
sample without a camera, API, or audio:

```sh
PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/check_sensors.py --port /dev/ttyUSB0 --samples 3
```

The parser validates protocol identity, request/boot/sequence correspondence,
channel types, bounds, and freshness. A timeout, disconnect, wrong firmware,
or malformed matching response produces unknown/unavailable channel feedback.
No echo is also unknown; it never means there is no obstacle.

## Pi keyboard session

Stop any other camera command first. Start the desktop window without speech:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase4.yaml --continuous --scene-live --preview --proximity --sensor-port /dev/ttyUSB0 --report artifacts/phase4.json
```

Add `--speak` after confirming the selected headphone output and comfortable
volume. Omit both `--proximity` and `--sensor-port` for camera descriptions
without the USB sensors. Configuration alone opens no serial port.

| Key in the focused preview | Action |
| --- | --- |
| D | Request one current unannotated camera image from Gemini; optionally narrate the result with fresh independent proximity feedback |
| P | Request fresh local proximity feedback; no image upload or cloud dependency |
| S | Stop local speech and invalidate pending narration |
| R | Rearm failed cloud actions and reconnect a paused sensor reader; does not upload by itself |
| q or Esc | Stop the session and release resources |
| Ctrl+C in the terminal | Cancel with the existing interruption behavior |

The Pi profile requests a 640×480 feed at 30 FPS with object detection off.
Camera startup includes one second of settling and one supported autofocus
cycle with a four-second wait limit, within the 30-second startup deadline.
This prepares the one-shot path too; it is not repeated for every D action.
It reuses `gemini-3.5-flash-lite`, a 60-second scene request timeout, and the
existing saved key/free-tier scope. Starting the window, P, S, R, or merely
having a saved key does not upload an image. Each accepted D requests one
image; the cloud path has one request in flight and a minimum five-second
cooldown after completion. Busy/cooldown key presses are ignored rather than
queued. A cloud failure pauses further D requests until R; local P remains
available without network access.

The left pane is current raw camera video; the right pane is the exact selected
scene image with its pending/unavailable/result status and age. A delayed
description refers to that snapshot, not the newest live image. S or quitting
prevents a late response from starting cancelled speech; an already active
HTTP request may still finish. No continuous narration or video recording is
started. Reports retain the latest 120 action records plus full-run counters;
images, reports, and other generated artifacts belong in ignored `artifacts/`.

## Single snapshot and Mac simulation

The Pi profile's default preview is off for this one-shot command:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase4.yaml --describe --proximity --sensor-port /dev/ttyUSB0 --speak --report artifacts/phase4-snapshot.json
```

Capture closes before the cloud description. Local proximity is acquired
after the scene work so an old sensor pair is not held throughout a slow
request. Scene text and channel observations remain separate report fields.
If the scene is unavailable, local sensor status can still be reported.

For Mac development, the supplied profile uses a synthetic camera frame and
invented proximity fixtures, visibly marked simulated:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv/bin/python -m vidi --config config/phase4-sim.yaml --continuous --scene-live --preview --proximity --report artifacts/phase4-sim.json
```

P uses only the local fixture; D still explicitly requests a Gemini upload
and requires the existing SDK/key. Do not press D during an offline-only
fixture check. Optional `--speak` uses native Mac `say`; no Pi dependency,
physical sensor, or new voice download is required for simulation.

## Feedback and failures

The provisional nearby threshold is **1000 mm**. Each channel expires at
**0.5 seconds**, using conservative host timing and device observation age.
Polling is capped at 5 pairs/s; the default response deadline is 0.4 seconds
and startup handshake limit is 3 seconds. Missing, stale, no-echo, out-of-range,
or failed observations have null distance and stay unknown. One working
channel does not make the other channel valid.

Feedback says “Left sensor: nearby return,” “return beyond the nearby
threshold,” or “reading unavailable,” separately for each channel. It never
assigns an ultrasonic distance to an object named by Gemini, invents a center
measurement, announces a safe direction, or estimates collision time.

`--speak` enables one bounded local utterance per accepted action, with sensor
feedback kept separate from an age-labelled camera description. The complete
scene result remains in the report even if the speech copy is shortened or
audio fails. Freshness is checked at the start of narration; the readings are
**a snapshot, not continuously current throughout the spoken utterance**.
Speech completion alone does not prove audibility. Existing OCR/read-aloud and
repeated Gemini detection commands keep their separate opt-ins and profiles.
Piper/model downloads remain subject to their pending separate approval.

## Controlled Phase 4 bench still to run

1. Record the actual ESP32 board/core/build/upload, divider values and echo
   voltage, USB path, firmware handshake, and physical left/right placement.
2. Test one sensor at a time, then both together, with a large flat stationary
   target at **0.5, 0.8, 1.2, and 1.5 m**. Collect **50 attempts per position
   per channel**. Record reference distance, valid-return fraction, median
   absolute error among valid readings, wrong-channel returns, and unknowns.
   Initial bench targets are **at least 90% valid** and **median absolute error
   at most 10 cm** in those controlled scenes; these are not measured results
   or real-world coverage guarantees.
3. Remove or move the target closer suddenly; test a missing sensor, unplugged
   USB, reconnect/reboot, invalid protocol, and stale data. Unknown must never
   become clearance; a stale channel must expire within the 0.5-second policy.
4. Capture a fresh camera-only baseline, then repeat with proximity, requested
   D, and optional speech under the same settings. Aim to retain **at least
   80% of baseline preview FPS**, with no blocking UI or accumulating queues.
   Record the actual timing windows; this target has not been measured.
5. Verify P with network unavailable, D failure followed by explicit R/D,
   repeated-key suppression, S cancellation, and q/Esc/Ctrl+C cleanup. Confirm
   audible output separately and rerun the working OCR/Gemini live commands.

Six native firmware tests passed by compiling the actual sketch against
temporary local SDK stubs; host tests use simulated serial/camera/cloud/audio
boundaries. They cannot verify dual-core ISR timing, IRAM placement, wiring, acoustic crosstalk,
accuracy, or physical playback. The successful ESP32 cross-compile is recorded
separately above; flashing and physical bench checks remain pending. Stop
directional/nearby performance claims if identity, freshness,
failure handling, or the controlled target checks fail. Battery runtime,
wearable mounting, sustained power/thermal optimization, and field evaluation
remain Phase 6; no mobility or navigation-safety guarantee is made.
