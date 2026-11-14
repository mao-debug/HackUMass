# vidi Phase 3: local reading and optional speech

Phase 3 reads printed text from one final camera/image frame using local
Tesseract. A separate explicit action can speak a short excerpt or save a WAV.
Camera resources close before OCR and speech begin. No cloud request, API key,
microphone, object-detection model, or Python OCR/TTS wrapper is required.

The connected target remains the existing Pi 4 with Camera Module 3/IMX708,
Debian 12, Python 3.11.2, Picamera2 0.3.27, and OS OpenCV 4.6.0. Preserve this
working installation. Phase 2 camera/person checks are recorded in
[hardware.md](hardware.md); the longer detection bench and broader chair
checks are independent of trying OCR. **Pi OCR, speaker/headphone operation,
and live speech have not been verified.** Phase 3 package approvals are pending.

## Explicit actions

| Option | Result |
| --- | --- |
| `--read-text` | Run OCR once on the final unannotated frame, after capture closes |
| `--read-text --speak` | Read locally, then request one short spoken response |
| `--read-text --speech-output FILE.wav` | Read locally, then synthesize one WAV without playing it |
| `--report PATH.json` | Save full recognized text and separate OCR/speech results |
| `--output PATH.png` | Save the final local image for checking framing and readability |

`--speak` and `--speech-output` require `--read-text` and are mutually exclusive.
Reading mode requires detector `none` and cannot be combined with `--describe`.
Configuration and installed engines alone never start OCR or speech. The
default sample run and existing detection commands retain their behavior.
One CLI invocation is the initial trigger; voice activation remains Phase 5
and the optional pushbutton remains Phase 6.

## Dependency setup: approval is pending

The Mac has native `say`, but Tesseract is absent. Approval is pending for
Homebrew Tesseract on Mac and these Pi OS packages:

```text
tesseract-ocr
tesseract-ocr-eng
espeak-ng
```

The following installation commands are reference steps **only after the
corresponding package installation is approved**. They have not been run as
part of Phase 3:

```sh
# Mac, after Homebrew Tesseract approval.
brew install tesseract

# Pi, after approval for these OS packages.
sudo apt install tesseract-ocr tesseract-ocr-eng espeak-ng
```

Do not reinstall desktop Python requirements on the Pi, replace its NumPy/
OpenCV/Picamera2 packages, or perform a blanket OS upgrade. Existing OS packages
are used directly through `/usr/bin/python3` with `PYTHONPATH=src`. Tesseract
and eSpeak NG are local executables; no `pytesseract`, `pyttsx3`, or cloud SDK
needs to be added for this phase.

After approved installation, these checks inspect local executables/language
data without recording images or playing audio:

```sh
tesseract --version
tesseract --list-langs
espeak-ng --version
```

Confirm that `eng` appears in the language list. On Mac, only run the eSpeak NG
check if that engine is separately installed; native `say` is used for live
speech there. The Pi's audio device, output routing, and comfortable volume
still require a controlled check with the user.

## Pi snapshot and OCR

From the repository root, after the dependencies are available:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi-ocr.yaml --read-text --output artifacts/pi-ocr.png --report artifacts/pi-ocr.json
```

`config/pi-ocr.yaml` selects:

| Setting | OCR profile |
| --- | --- |
| Camera/source | `picamera2`, camera index 0 |
| Requested image size and FPS | 1280×960 at 5 FPS |
| Settling interval | 1 second after camera startup |
| Autofocus | One cycle if `AfMode` and `AfTrigger` controls are available |
| Autofocus wait limit | 4 seconds |
| Startup/capture deadlines | 30 seconds / 5 seconds |
| Final frames and detector | 1 / `none` |
| OCR/TTS engines | `tesseract` / `espeak_ng` |

The worker returns an owned, unannotated BGR frame and closes before OCR.
Unsupported autofocus controls or an unsuccessful completed focus cycle
produce a warning and still allow a snapshot. An autofocus timeout or
unexpected focus error cancels pending work and aborts capture instead of
queuing a snapshot behind a stuck focus job. A camera-reported focus success
does not establish that the printed text is readable. The implementation uses
the [versioned Picamera2 0.3.27 autofocus/wait APIs](https://github.com/raspberrypi/picamera2/blob/v0.3.27/picamera2/picamera2.py).

Use a stationary, well-lit printed English sign or page with large text for
the first check. Keep the full text in view and inspect the saved image.
Small characters, blur, glare, and tilted text can reduce recognition quality;
see Tesseract's [image-quality guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html).
Higher resolution and autofocus are preparation steps, not accuracy guarantees.

## Speech or WAV export

Start with a WAV if the Pi output device is not yet confirmed. This synthesizes
locally and does not initiate playback:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi-ocr.yaml --read-text --speech-output artifacts/pi-reading.wav --report artifacts/pi-reading.json
```

After verifying the output device and comfortable volume, request one spoken
response:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src /usr/bin/python3 -m vidi --config config/pi-ocr.yaml --read-text --speak --report artifacts/pi-speaking.json
```

Speech defaults to 150 words per minute and a 60-word text excerpt. A truncated
excerpt adds “More text is available.” The JSON report retains the complete
recognized text. A successful OCR run with no text uses the short feedback
“No readable text found.” OCR failures remain errors rather than invented text.
Speech has a 45-second process timeout; use Ctrl+C to interrupt the command.
There is no repeating narration loop.

The Pi profile uses eSpeak NG's English voice (`en`). A Mac run with automatic
engine selection uses native `say` and Samantha. WAV export uses eSpeak NG;
the native `say` backend is for live speech in this implementation. The eSpeak
NG project documents [local command-line operation](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md).

## macOS image development

Use the existing VS Code `.venv` and desktop dependencies from the README.
After approved Tesseract installation, replace the example path with a local
image containing printed text:

```sh
python -m vidi --image path/to/printed-text.png --detector none --read-text --report artifacts/mac-ocr.json
python -m vidi --image path/to/printed-text.png --detector none --read-text --speak
```

These image commands need no Pi camera or microphone. Do not use the Pi camera
profile on Mac. A Mac speech result does not verify the Pi audio path.

## Configuration and result interpretation

OCR defaults are `ocr.language: eng`, `psm: 6`, `preprocess: grayscale`,
`timeout_s: 15.0`, and `executable: tesseract`. The `none` preprocessing option
keeps the color image; preprocessing operates on a copy. Tesseract receives
local image bytes and returns local text. PSM 6 assumes one text block; use the
supported page-segmentation settings for a different layout after testing.
See the [Tesseract command-line guide](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).

Speech settings are `speech.speed_wpm: 150`, `max_words: 60`, `timeout_s: 45.0`,
and an optional `voice`. Backend selections live in `models.ocr_backend` and
`models.text_to_speech_backend`. CLI action flags are still required. YAML
paths resolve relative to their configuration file; CLI paths resolve relative
to the terminal directory. Save reports, images, and audio in ignored
`artifacts/`; keep exported files distinct.

An OCR result distinguishes recognized text, no readable text, and failure.
Speech has its own result. **Speech status `completed` means the engine
finished, not that anyone heard the audio.** A WAV file is not a playback test.
Reports add `ocr` and `speech` sections to the existing JSON fields. OCR statuses
are `not_requested`, `ready`, `no_text`, `failed`, or `cancelled`. Speech statuses
are `not_requested`, `pending`, `skipped_ocr_failed`, `completed`, `failed`, or
`cancelled`. The complete OCR text is checkpointed before printing or starting
speech; report updates replace the previous JSON atomically. A speech failure
returns exit status 1 while preserving recognized text. Ctrl+C returns 130 and
attempts to record cancellation without discarding the preceding checkpoint.
Missing executable/language data, process failures, and timeouts must be
reported as such. Neither an empty OCR result nor any spoken text establishes
that navigation is safe.

## Controlled acceptance still to run

After package approval and installation, record the actual Pi command,
requested/delivered image dimensions, printed ground truth, recognized text,
OCR latency, and artifact paths. Then check an empty scene, an image with more
than 60 words, camera/process failures, and interruption cleanup. Confirm
WAV synthesis separately from speaker/headphone audibility. Use the same
controlled page with network access unavailable to check the offline path.

Software tests exercise subprocess and hardware boundaries with local fakes,
including real harmless child-process timeout and cancellation checks. Optional
real-engine tests generate English sign images for Tesseract and synthesize a
WAV with eSpeak NG without playback. They explicitly skip when the respective
engine is absent; no engine is downloaded by a test or application startup.
Current Mac validation: **486 passed, 4 skipped**, running outside the restricted
sandbox so the fake-camera shared-memory test could execute. The four skips
are the three Tesseract examples and one eSpeak NG WAV check because those
engines are not installed. Dependency consistency and whitespace checks passed.
No new on-device OCR or audio acceptance result is claimed here. This prototype
supplements established mobility aids and does not provide precise distance
measurements or guaranteed navigation safety.
