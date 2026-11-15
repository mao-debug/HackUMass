# vidi hands-free assistant

`--assistant` runs independently of a desktop window. The camera and microphone
stay active locally; **“hey vidi” (VEE-dee)** activates one command clip of at
most five seconds. Wait for the short cue before speaking the command. Repeat
the wake phrase before each question. Idle audio stays in bounded RAM, with no
background audio/video files or uploads. Wake detection continues during speech.

The existing GUI demos remain available. Actual Pi wake accuracy, simultaneous
Bluetooth audio, interruption latency, chess accuracy and cold boot still need
physical acceptance; the new software does not establish those results.

## Setup

First update the Pi checkout to the published Phase 6 feature branch:

```sh
cd ~/vidi
git remote set-branches --add origin feat/vidi-phase6
git fetch origin
git switch feat/vidi-phase6
git pull --ff-only
```

Keep Raspberry Pi OS NumPy, OpenCV, Picamera2 and libcamera in place. Use
[wake setup](wake.md) and [voice auditions](voices.md) for the approved exact
packages and model sources. Piper uses a separate `.venv-tts`. The user selected
**LJ**, so the Phase 6 profile uses `en_US-ljspeech-medium`. Install its isolated
runtime and voice on the Pi before running that profile. Existing profiles
retain eSpeak for comparison. Gemini speech additionally needs
`--cloud-speech` and sends response text to Google.

The local chess package is approved for the Pi:

```sh
cd ~/vidi
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m pip install --no-deps chess==1.11.2
PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/check_assistant.py --config config/pi-phase6.yaml
```

Plug in the USB recovery keyboard. Copy its actual `available_keyboard_links`
entry ending in `event-kbd` into `assistant.keyboard_device_path` in the YAML.
The ordinary Pi user must be able to read that selected device. The app does
not grant permissions or run as root; review the exact permission change if
access is denied. Do not make input devices world-readable. Discovery does
not verify hardware operation. A monitor is used for setup/calibration only.

The ESP32 may remain disconnected. The new profile disables proximity. After
the soldered assembly returns and is checked, configure the serial backend
and port explicitly before adding `--proximity`.

## Run

Mac keyboard rehearsal uses synthetic pixels and invented chess positions:

```sh
PYTHONPATH=src .venv/bin/python -m vidi --config config/assistant-sim.yaml --assistant --mode chess --offline --no-wake --preview --run-seconds 60 --report artifacts/assistant-sim.json
```

It is explicitly simulated, without microphone or Gemini requests. Initial Pi
keyboard checks can also use `--no-wake --preview`. With local wake assets and
the recovery keyboard ready, the foreground Pi command is:

```sh
cd ~/vidi
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase6.yaml --assistant --mode normal --cloud --speak --report artifacts/assistant.json
```

| Spoken command | Result |
| --- | --- |
| “What is directly in front of me?” / “describe” | One fresh current-view Gemini image; speaks snapshot age |
| “Read the text in front of me” | One requested Gemini OCR image |
| “Capture an image” / “take a photo” | Saves an unannotated local photo, without uploading |
| “Proximity” | Independent fresh local sensor feedback if enabled |
| “Chess mode” / “normal mode” | Switches modes, preserving confirmed game history |
| “What moved?” | Requested chess scan |
| “Whose turn?” / “what is on e four?” | Confirmed-board information after a fresh scan |
| “Confirm position” / “reject position” | Separate manual confirmation/rejection |
| “Repeat” / “help” / “stop” | Bounded reply, help or cancellation |
| “Stop listening” | Closes camera/mic and clears transient audio; F12 resumes |

Only documented whole commands are accepted. The camera sees its current
field of view, not the entire area around or behind the wearer. Free-form
contextual follow-ups and automatic outdoor/sign/crosswalk/approach alerts
remain the next milestone.

At most one cloud request is active. Requested image scans are spaced by at
least five seconds after the previous image request completes. Fresh images
are selected after that wait. Command transcription does not add an artificial
image cooldown. Errors pause uploads until explicit R recovery. A cancelled
network request may finish, but its result cannot narrate or change game history.

## Physical recovery keys

Without a window, the selected USB keyboard provides **Space** stop,
**F12** camera/mic privacy pause/resume, and **Esc** quit. **R** rearms requests
and retries devices; **H** gives help. **D/T/P** request scene/text/proximity;
**B/C/X** scan/confirm/reject; **1/2** select normal/chess; **M** requests a
command clip. No typed text is logged and the keyboard is not grabbed.
Losing the selected keyboard or microphone causes a privacy pause.

The preview supports the same letter keys, Space, Q/Esc and platform F12.
Switching back to chess requires a new scan before board queries. Wake preserves
an existing manual proposal for an explicit confirmation command until expiry;
Space, mode changes, privacy pause and quit discard unconfirmed proposals.

## Chess

Use the existing board and [four printed markers](chess.md#experimental-calibration-and-automatic-tracking).
A sighted setup partner calibrates its corners in a saved image. All four
markers must remain fixed to the board plane and visible with the whole board.
Follow the [chess guide](chess.md) for marker generation and calibration.

Automation stays disabled until the physical acceptance record passes. Its
explicit opt-in command is:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase6.yaml --assistant --mode chess --cloud --speak --automatic-chess --calibration artifacts/chess-calibration.json --report artifacts/assistant-chess.json
```

Each requested scan acquires two distinct stable frames at least 300 ms apart,
rectifies each from its own four-marker geometry, and sends both in one request.
Both complete 64-square predictions must agree. Local rules accept an unchanged
position or exactly one legal half-move from confirmed history, starting from
the standard initial layout. Unknown, stale, moved, missing-marker or inconsistent
views abstain. Scan after both players' moves. Agreement and legality do not
prove visual accuracy. The original `--chess-live` mode remains human-confirmed.

## Startup without a terminal

`deploy/vidi-assistant.service` is a reviewable **user service template**; it has
not been installed or enabled. It defaults to manual chess. Confirm assets,
keyboard access, headphones, the existing private saved key and the foreground
command first. The template waits for user audio services, uses bounded restart
attempts, and never embeds the API key. After approving deployment on the Pi:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/vidi-assistant.service ~/.config/systemd/user/vidi-assistant.service
systemctl --user daemon-reload
systemctl --user enable --now vidi-assistant.service
journalctl --user -u vidi-assistant.service -n 40 --no-pager
```

Starting before login additionally needs an approved user-session/linger setup
and Bluetooth/audio readiness check. Enabling a user unit does not establish
cold-boot audio. Stop it and restore foreground operation with:

```sh
systemctl --user disable --now vidi-assistant.service
```

Pass two actual cold boots without a terminal, full-duplex audio, recovery keys
and a 15-minute combined run before marking deployment complete. See the
[Phase 6 acceptance record](phase6.md).
