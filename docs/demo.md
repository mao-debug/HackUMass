# vidi: three-minute HackUMass demonstration

vidi is a smart glasses prototype that provides requested scene descriptions,
printed-text reading, and separate local left/right ultrasonic feedback. It
supplements established mobility aids. Do not present it as precise object
distance measurement, guaranteed obstacle coverage, or safe navigation.

The user has heard the Pi's scene description and text reading. The supplied
voice log accepted `describe`; it does not establish that OCR was triggered
by voice. This demonstration therefore uses **M for description and T for
reading**. Recognition accuracy, measured sensor accuracy, sustained
performance, and remaining hardware upgrades are deferred to the
[Phase 6 plan](phase6.md).

## Prepare when back at the Pi

Use the existing Pi 5 installation, saved key, Bluetooth headset, IMX708
camera, and protocol-v1 ESP32 with the handoff checkout when available.
Preserve the working installation and local files.

1. Connect the working camera/ESP32 setup and select the headset as both
   desktop microphone input and playback output. Keep a comfortable volume.
2. Close other camera applications and Arduino Serial Monitor. Work from
   `~/vidi`; the retained interpreter is `.venv-pi/bin/python`.
3. Check the current serial path if needed:

   ```sh
   cd ~/vidi
   PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/check_sensors.py --list
   ```

   The observed path was `/dev/ttyUSB0`; replace it below if enumeration
   changed. Listing does not open the serial device.
4. Set a stationary, non-sensitive scene in view and have the printed card
   below ready. Use the existing approved Gemini key/project; do not change
   billing or install new packages for the presentation.
5. Start the full Pi session with explicit cloud and voice options:

   ```sh
   python3 scripts/run_demo.py --target pi --cloud --voice --proximity --sensor-port /dev/ttyUSB0 --speak --run
   ```

Without `--run`, the launcher prints only the plan and opens nothing. It
selects the existing `.venv-pi/bin/python`, the Pi profile, and a unique report
filename under ignored `artifacts/`; the actual path is printed. It installs
nothing and does not change routing or record a video stream. The underlying
manual command remains in [Phase 5](phase5.md).

The camera profile requests 640×480 at 30 FPS; that is not a guaranteed rate.
Keep the preview focused. Each accepted M captures up to five seconds, and
Gemini stages share a five-second cooldown after each completed cloud call.
Wait for an idle/ready state before the next action; repeated busy key presses
are ignored rather than queued. The image-age caption describes the selected
snapshot, not total time to hear an answer.

## Presentation sequence

These are rehearsal time budgets, not cloud-latency guarantees. If a request
is slow or unavailable, use the local fallback below instead of repeatedly
triggering it.

| Time | Action | Explain or show |
| --- | --- | --- |
| 0:00–0:25 | H | Requested local help. “vidi gives information when asked; it does not continuously listen or narrate.” |
| 0:25–0:50 | P | Separate left/right local sensor status. “A nearby return belongs to a sensor channel, not a distance to an object named by the camera.” |
| 0:50–1:35 | M, then say “Describe the scene” | One short clip, Gemini transcription, the accepted command, and one selected camera snapshot. Show current camera pixels separately from the described image and listen to the response. |
| 1:35–2:20 | Hold up the printed card, then T when idle | One explicit image OCR request and local read-aloud. Use the keyboard trigger so the demo does not imply voice-triggered OCR was previously validated. |
| 2:20–2:40 | S during an utterance | Stop local narration. If the text response has already finished, H provides another short local utterance to stop. |
| 2:40–3:00 | Q or Esc | Release resources. Close with the demonstrated capabilities and the remaining accuracy/power/wearability work. |

Full controls: **D** requests a scene directly; **P** local proximity; **M**
one command clip; **T** image OCR; **H** local help; **S** stop/cancel narration;
**R** rearm failed paths/reconnect sensing; **Q/Esc** quit. Ctrl+C cancels from
the terminal. R does not upload by itself. Spoken “stop” requires M and
transcription; S/Q are the immediate local controls.

No recorded clip or raw transcript is saved in the report. Requested scene/
OCR text and sensor/status metadata can be saved. A spoken sensor statement
is checked at narration start and is a snapshot, not continuously updated
throughout the utterance.

## One printed test card

Copy these ASCII words into an editor and print them in large black letters
on plain matte white paper, with generous margins. Use a large font such as
48–72 points, keep the card stationary and well lit, and keep all lines in view.

```text
VIDI DEMO
WELCOME TO HACKUMASS
ROOM 101
```

Compare the returned words with the card. This is a presentation prop, not a
completed OCR accuracy benchmark. If small print needs a separate high-resolution
snapshot, the [Phase 3 profile](phase3.md) remains available after closing this
camera session.

## If network access is unavailable

Quit the cloud session and use the explicit offline guard:

```sh
python3 scripts/run_demo.py --target pi --proximity --sensor-port /dev/ttyUSB0 --speak --run
```

This keeps the live camera and **H/P/S/R/Q** local, disables **D/T/M**, and
does not load saved credentials. M transcription and D/T scene/text requests
require explicit cloud mode and Gemini connectivity. An unavailable request
is not a successful empty result.
Do not describe offline Tesseract as working: its installation/acceptance is
still pending. After an error, R only rearms the path; a new explicit action
is needed to retry when connectivity returns.

No presentation video is currently bundled or linked. On a later Pi visit,
the team may record a short, non-sensitive successful H/P/M/T/S/Q run using
an available local screen recorder or a camera. Save and check that recording
before relying on it, and label any playback **prerecorded** with its date.
It can illustrate the online capabilities while the live local controls are
shown; it does not establish fresh recognition or network availability.

## Rehearse away from the Pi

First inspect the default plan, then launch a local rehearsal if wanted:

```sh
python3 scripts/run_demo.py
python3 scripts/run_demo.py --proximity --run
```

The first command only prints a plan; the second opens synthetic pixels and
proximity fixtures in offline mode. **D/T/M are blocked** and credentials are
not loaded. Rehearse **H/P/S/R/Q**; add `--speak` for requested native Mac `say`
output. This does not access Pi hardware.

Cloud features require `--cloud`; M additionally requires `--voice` and, in
simulation, a valid local `--command-audio FILE.wav`. No audio asset is bundled.
`--python PATH` can select another existing interpreter; `--artifacts-dir DIR`
changes the directory for unique reports. `--prompt-key` is available only
with cloud mode, and Pi voice can use a literal `--microphone-device NAME`.
With `--run`, the app stays in the foreground so normal cancellation applies.
See [Phase 5](phase5.md) for WAV limits and [Phase 6](phase6.md) for deferred
optimization work.
