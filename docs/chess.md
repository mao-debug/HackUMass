# Experimental chessboard information

The default optional mode follows a **human-confirmed chess position**. B requests one
board observation; it can propose the standard starting position or one legal
move from the last confirmed position. A separate C confirms the proposal only
after a person checks the physical board. It reports board information, not
recommended moves, engine analysis, or autonomous play.
The separate opt-in automatic path below can commit a matching pair of board
observations through the same legal-history checks. It remains experimental
and is disabled by default; two agreeing predictions do not establish accuracy.

**Real board recognition from the glasses has not been validated.**
The offline fixture demonstrates the software workflow;
it is not evidence that a camera can reliably identify a physical board,
orientation, pieces, or moves. Do not rely on it for blindfold play before
checking actual board images with a sighted partner.

## Optional package and scope

The local rules/state dependency is **chess 1.11.2**, also available through
the project's `chess` extra. Its download into the Mac `.venv` was approved
and completed. Existing default/help/general modes do not require or import
the optional rules package. The package's board/legal-move interfaces are
described in [python-chess core documentation](https://python-chess.readthedocs.io/en/latest/core.html);
the pinned distribution is [chess 1.11.2](https://pypi.org/project/chess/1.11.2/).
No Stockfish, chess engine, extra AI provider, or engine download is needed.

Pi installation is approved and remains to be run. The existing camera
environment can add the local dependency without replacing its OS packages:

```sh
PYTHONNOUSERSITE=1 .venv-pi/bin/python -m pip install --no-deps -r requirements-chess.txt
```

This is a reference command, not an installation already performed on the Pi.
Preserve its OS camera/OpenCV stack and existing Google SDK/key. This mode
does not use ultrasonic sensors; use the general scene mode for proximity.

## Rehearse offline on Mac

From the repository root, the launcher prints a plan unless `--run` is present:

```sh
python3 scripts/run_demo.py --chess
python3 scripts/run_demo.py --chess --speak --run
```

This selects `config/chess-sim.yaml`, synthetic pixels, and explicitly
simulated board fixtures. It never loads credentials or uploads images/audio.
Cloud and voice flags are rejected for this fixture launcher. Optional speech
uses the existing local Mac engine; every fixture response is marked
**SIMULATED**. The six invented positions are standard start, e4, e5, Nf3,
Nc6, Bb5. These are prescribed fixture examples, not suggested moves.

Press B, inspect the proposed starting position, and press C. Repeat B then C
for each subsequent half-move. B only observes; it does not commit a move.
With speech enabled, wait for the proposal utterance to finish before pressing
C; a C press during speech is ignored rather than queued. Resolve each pending
proposal with C or X before requesting another scan.
The general scene demo remains separate and unchanged.

## Prepare a real board after returning to the Pi

1. Start from the **standard starting position**, with White to move. The mode
   does not initialize an arbitrary middle-game position or infer missing game
   history from a picture.
2. Keep a fixed camera view with **all 64 squares and all four corners visible**.
   Remove hands before each scan. Start with a stable mount/view and clear
   lighting; a moving glasses camera is not a validated setup.
3. Know the orientation before scanning. `white_near` means White's home side
   is nearest the camera; `black_near` means Black's side is nearest. Put clear
   coordinate corner labels around the board and have a sighted partner check
   a1/h1/a8/h8 against the view. There is no claimed automatic pose calibration.
4. Compare the actual first observation with every piece in the known starting
   position. The partner presses C only after verification. If it is unclear
   or wrong, press X and correct the setup before a fresh B.
5. After **every half-move, including your own**, remove hands, press B, check
   the proposed change against the physical board, then separately confirm C
   or reject X. Do not skip moves and assume the internal history caught up.

After Pi package setup, use this session for controlled initial board-image
checks:

```sh
python3 scripts/run_demo.py --target pi --chess --cloud --voice --speak --run
```

Omit `--voice` for keyboard-only operation. The launcher requires `--cloud`
for Pi chess and rejects proximity flags. Starting the window does not upload
a board: B is the explicit scan action. Existing key/free-tier scope is reused;
there is no billing change or new provider. Each requested M can upload one
short command clip; a recognized scan command can then upload one image.

For a known orientation override, use the manual CLI:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-chess.yaml --chess-live --continuous --preview --board-orientation black_near --speak --report artifacts/chess-pi.json
```

`config/pi-chess.yaml` requests Picamera2 1280×960 at 15 FPS, one settling/
supported-focus preparation, detector `none`, the existing Gemini model with
a 60-second cloud timeout, and a maximum uploaded image edge of 1024 pixels.
Requested FPS/resolution are not measured recognition or performance results.
Speech is locally bounded to 100 words and a 60-second timeout; a spoken
excerpt need not enumerate the entire board. The launcher prints a unique
report path under ignored `artifacts/`; it does not record a video stream.

## Controls and voice

| Key | Action |
| --- | --- |
| B | Request one new board observation and propose, without confirming |
| C | Confirm the current proposal after checking the physical board |
| X | Reject/discard the proposal |
| L | Read the last confirmed move |
| W | Read whose turn follows from confirmed history |
| A | Summarize the confirmed board |
| H | Local help |
| S | Stop speech, cancel pending work, and discard an unconfirmed proposal |
| R | Rearm a failed observation/voice path; no upload by itself |
| Q/Esc | Quit; Ctrl+C also cancels from the terminal |
| M, if enabled | Record one requested command clip of at most five seconds |

Supported complete chess phrases after M include “check board,” “confirm
move,” “reject move,” “whose turn,” “last move,” “describe the board,” “what is
on e4,” “repeat,” “stop speaking,” and “help.” These report or confirm state;
spoken move orders are not executed. Voice requires Gemini chess and cannot
be enabled in fixture mode. M transcription is online even for a query that
can otherwise be answered locally; keyboard L/W/A/H avoid that audio upload.
S/Q are immediate local controls, not spoken wake-word commands.

The live image and exact scanned image remain separate. Failed, incomplete,
ambiguous, or old observations must not advance confirmed state. Proposals
expire under `chess.observation_max_age_s` (default 30 seconds; maximum 120).
If a proposal has expired, discard it with X and scan again instead of
confirming old visual data. Square/turn queries identify the last confirmed
position and mention when a newer observation still awaits confirmation.
R plus a new B is required after an observation failure; retries are not
automatic. Confirmation is a distinct person-requested action, not a result
of asking the camera to look.

## Configuration, boundaries, and acceptance

The strict `chess` YAML section has `backend` (`gemini` or `fixture`), optional
`fixture_path`, fixed `orientation`, and `observation_max_age_s`. CLI
`--chess-fixture PATH` selects fixture observations; `--board-orientation`
overrides orientation. Both require `--chess-live`. That mode requires
`--continuous --preview`, an ordinary camera source, and detector `none`.
It cannot be mixed with scene/detection/OCR modes, `--question`, proximity, or
WAV speech export. `--speak` is optional local playback.

`--offline` chess is valid only for fixtures. Fixture chess always bypasses
credential helpers, even without that flag, and rejects key prompts and voice
capture. Gemini chess permits the existing optional hidden `--prompt-key`.
Exports cannot overwrite fixture, camera-image, or voice-audio inputs.
The optional rules package is imported only when the chess session is started.

Software/fixture tests do not establish physical recognition. The next
device-dependent check is a labelled, known starting board followed by single
verified moves, under the intended full-board view and orientation. Record
wrong/unknown squares and rejected proposals rather than claiming accuracy
from a plausible move alone. Actual Pi package setup, physical images, partner
confirmation, and glasses-camera usability remain pending. Existing deferred
hardware accuracy, power, sustained performance, and wearable work remain
Phase 6; this prototype supplements established mobility aids.

## Experimental calibration and automatic tracking

Automatic tracking is a separate explicit opt-in, disabled in existing profiles.
It has **no physical chessboard acceptance result**. Retain the human-confirmed
workflow until real board images, orientation, piece identities, and special
moves have been checked with a sighted partner. The software does not choose
moves, move pieces, or run an engine.

The geometry step uses four different local ArUco `DICT_4X4_50` markers:
**0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left**, relative to the
camera's board view. Print the generated sheet, cut out each labelled marker
with its white margin, and place them outside the board corners on the same
fixed flat surface. Keep marker centers outside every board corner and leave
all four markers visible. Coordinate labels and a verified `white_near` or
`black_near` orientation are still required; marker detection does not infer
chess coordinates or piece identities.

Generate the printable SVG locally, without downloading any marker assets:

```sh
PYTHONPATH=src .venv/bin/python scripts/calibrate_chess.py --markers artifacts/chess-markers.svg
```

These tools need an OpenCV build with `cv2.aruco`. Both OpenCV 4.6 legacy and
new detector APIs are supported. If the Pi's existing build lacks ArUco,
calibration fails clearly: keep this option disabled and obtain approval for
an appropriate package change instead of replacing the working camera stack.
No package is installed by this script.

After returning to the Pi, capture one local setup image with the full board,
all four markers, hands removed, and the final camera view. The capture below
has no Gemini request or speech:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-chess.yaml --detector none --max-frames 1 --output artifacts/chess-calibration.png
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python scripts/calibrate_chess.py --image artifacts/chess-calibration.png --output artifacts/chess-calibration.json --orientation white_near
```

In the calibration window, click the board's outer **top-left, top-right,
bottom-right, bottom-left** corners, then press Enter. X clears the clicks;
Q/Esc cancels. Click the actual 8×8 board boundaries, not marker corners.
The bounded local JSON records the clicked board relative to all marker
corners and the fixed orientation. Existing outputs are never overwritten.
Repeat calibration when the board-to-marker arrangement changes. Do not
commit personal setup images/calibration from ignored `artifacts/`.

For every requested automatic scan, each of two distinct fresh source frames
must independently show all four unique markers. Missing/duplicate markers,
invalid corners, or marker positions inconsistent with one plane cause
abstention; there is no last-good crop fallback. The two frames are separated
by at least 300 milliseconds and rectified into a square image. One Gemini
request carries both rectified images. Each image must independently report
all 64 squares, known orientation, full visibility, and no unknown symbols;
the two complete layouts must agree exactly. A hand or piece motion, missing
square, disagreement, expiry, cancellation, or late scan result cannot commit
a change. A matching pair is still a correlated model prediction, not an
independent measurement of ground truth.

The automatic wrapper accepts only the standard initial layout and then one
uniquely legal half-move from confirmed history. Castling, en passant, and
promotion use that recorded history; skipped moves and arbitrary middle-game
initialization abstain. Both images expire within 30 seconds of capture,
including time spent in recognition. A pair that has already contributed to
history cannot be reused. Simulation remains explicitly marked SIMULATED.
Stopping invalidates the scan generation so a delayed result cannot rearm or
change state. Existing manual single-image scan/confirm behavior is preserved.

The new assistant runtime selects this path explicitly. A controlled,
60-second keyboard/preview bench after setup uses:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase6.yaml --assistant --mode chess --cloud --speak --automatic-chess --calibration artifacts/chess-calibration.json --keyboard-device /dev/input/by-id/USB_KEYBOARD-event-kbd --no-wake --preview --run-seconds 60 --report artifacts/phase6.json
```

Replace the keyboard path with the verified recovery keyboard on that Pi;
the example is not evidence that this device path exists. Headless operation
requires a selected recovery keyboard, even when wake input is enabled.
The initial bench disables wake input; wake assets/setup and real-board
acceptance remain separate physical checks. The Phase 6 profile uses the
user-selected local LJ Piper voice after its Pi setup. Existing profiles retain
eSpeak. Optional cloud speech requires its own configured backend and explicit
`--cloud-speech`; that choice is not needed to check board geometry.
The existing `--chess-live` GUI remains the manual B/C workflow; adding
automatic support does not silently change it. User approval covers requested
two-image chess batches, not automatic background image uploads.

Software checks cover synthetic marker planes, absent/duplicate markers,
legacy/new detector APIs, strict paired responses, disagreement, freshness,
cancelled generations, and ordinary/special legal moves. They are not a Pi
camera calibration, recognition accuracy, or wearable usability result.
