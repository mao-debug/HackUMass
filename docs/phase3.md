# vidi Phase 3: printed text and optional local speech

Phase 3 offers two explicit, one-image reading actions: `--gemini-read` sends
the final unannotated frame to Gemini for transcription, while `--read-text`
uses local Tesseract offline. Both close the camera before recognition begins.
Either can print the complete text, save a JSON report, and optionally speak
one short excerpt or export a WAV using a local speech engine. Gemini is used
for reading the image, not for generating audio.

The current target is a Pi 5 using the previous Pi 4 SD card. The recorded
Pi 4 baseline used Camera Module 3/IMX708, Debian 12, Python 3.11.2,
Picamera2 0.3.27, and OS OpenCV 4.6.0. Preserve the installation and follow the
[Pi 5 migration checks](pi5-migration.md) to verify the camera and audio on
the new board. Phase 2 camera/person checks are recorded in
[hardware.md](hardware.md); the longer detection bench and broader chair
checks are independent of trying OCR. The supplied Pi Gemini run returned
HTTP 200 and `OCR=ready` with printed text; that run did not request speech.
The user subsequently reported audible speech and described the voice as
robotic. Controlled headphone routing, complete read-aloud accuracy, and the
optional neural voice still need bench verification. Pi eSpeak NG installation
is approved; Tesseract package approval is still pending.
Gemini reading can use the existing SDK and saved key
without either local OCR or speech executable.

## Explicit actions

| Option | Result |
| --- | --- |
| `--gemini-read` | Send one final unannotated image to Gemini to transcribe printed text |
| `--read-text` | Run local Tesseract once, without a cloud request or API key |
| Either reading flag with `--speak` | Request one short response from the local speech engine |
| Either reading flag with `--speech-output FILE.wav` | Synthesize one local WAV without playing it |
| `--report PATH.json` | Save full recognized text and separate OCR/speech results |
| `--output PATH.png` | Save the final local image for checking framing and readability |

Choose exactly one reading flag. Reading requires detector `none` and cannot
be combined with detection, scene description, or continuous/live actions.
Playback and WAV export are mutually exclusive. These speech actions can also
accompany a separate `--gemini-detect` or `--describe` request; see
[gemini.md](gemini.md).
Configuration and installed engines alone never start OCR or speech. The
default sample run and existing detection commands retain their behavior.
One CLI invocation is the initial trigger; voice activation remains Phase 5
and the optional pushbutton remains Phase 6.

## Pi Gemini reading: start with text only

Stop the live preview first with q/Esc or Ctrl+C, and wait for that command to
exit so the camera is available. From the repository root, use the existing
`.venv-pi` containing the approved Google SDK and the saved API key:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini-ocr.yaml --gemini-read --output artifacts/pi-gemini-ocr.png --report artifacts/pi-gemini-ocr.json
```

This command prints recognized text and does not play audio. The profile
captures one 1280×960 image at a requested 5 FPS, allows one second to settle,
and requests one supported autofocus cycle with a four-second wait limit.
The camera closes before the single cloud request. The uploaded image is
resized to a maximum long edge of 1024 pixels. The profile uses
`gemini-3.5-flash-lite`, a 60-second OCR request timeout, and a maximum output
budget of 2048 tokens. These limits do not guarantee that an entire dense page
will be transcribed.

`--gemini-read` is the per-run upload opt-in; YAML and a saved key alone do not
send images. It is separate from repeated live identification and never starts
continuous reading. Use a controlled, non-sensitive printed test page and
check the transcription against it. See [Gemini setup](gemini.md) for saved-key
handling and the existing free-tier/no-billing-change scope, and Google's
[image-input documentation](https://ai.google.dev/gemini-api/docs/image-understanding)
for the cloud image path. No Tesseract installation or new Python dependency
is required for this command.

After the headphones are selected as output and eSpeak NG is available, add
`--speak` for one local spoken excerpt:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini-ocr.yaml --gemini-read --speak --report artifacts/pi-gemini-speaking.json
```

To export audio without playback, use `--speech-output` instead:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini-ocr.yaml --gemini-read --speech-output artifacts/pi-gemini-reading.wav --report artifacts/pi-gemini-reading.json
```

Both speech options need the local eSpeak NG executable on the Pi. The full
OCR result is saved before speech starts, so a missing engine or audio error
does not discard the transcription. No Gemini text-to-speech API is called.
An optional local Piper voice and camera-free audition are described in
[voice setup](voices.md); new runtime/model downloads require separate approval.

## Bluetooth headphones on the Pi

Pairing and audible output still need to be checked on the connected desktop:

1. Put the headphones into their pairing/discoverable mode.
2. Click the Pi desktop's Bluetooth icon, enable Bluetooth if needed, choose
   **Add Device**, select the headphones, and complete pairing.
3. Right-click the speaker/volume icon and select the headphones as the audio
   output. Unmute and start with a comfortable low volume. Pairing alone does
   not select the playback destination.

These steps follow Raspberry Pi's official
[audio setup guide](https://pip-assets.raspberrypi.com/categories/1259-audio-camera-and-display/documents/RP-008124-WP/Choosing-an-Audio-option.pdf)
and [audio output selector](https://www.raspberrypi.com/documentation/computers/configuration.html#change-audio-output).
No headphone microphone is needed for this phase.

Check speech-engine availability without playing anything:

```sh
command -v espeak-ng
```

The user's check printed no path: eSpeak NG is missing. Its approved installation
steps are below; completion and playback still need verification. If pairing or
routing is unclear, these read-only checks identify tools already present:

```sh
command -v bluetoothctl pactl wpctl
```

Run `bluetoothctl show` only if available. Use `pactl info` and
`pactl list short sinks`, or `wpctl status`, if those tools are present to
inspect the audio stack and outputs. Keep output selection in the desktop UI;
do not install or replace the audio stack just because one diagnostic command
is absent. An engine's successful exit or a generated WAV does not establish
that the headphones played sound.

## Local engine setup and approvals

The Pi's `espeak-ng` package and required dependencies from its configured
Debian/Raspberry Pi OS apt repositories are approved. Run these steps on the
Pi, then check the engine version; no installation has been run by this agent:

```sh
sudo apt update
sudo apt install espeak-ng
espeak-ng --version
```

The Mac has native `say`, but Tesseract is absent. Approval remains pending
for Homebrew Tesseract on Mac and these Pi OCR packages:

```text
tesseract-ocr
tesseract-ocr-eng
```

The following installation commands are reference steps **only after the
corresponding package installation is approved**. They have not been run as
part of Phase 3:

```sh
# Mac, after Homebrew Tesseract approval.
brew install tesseract

# Pi OCR only, after approval for these packages.
sudo apt install tesseract-ocr tesseract-ocr-eng
```

Do not reinstall desktop Python requirements on the Pi, replace its NumPy/
OpenCV/Picamera2 packages, or perform a blanket OS upgrade. Existing OS packages
are used directly through `/usr/bin/python3` with `PYTHONPATH=src`. Tesseract
and eSpeak NG are local executables; no `pytesseract`, `pyttsx3`, or cloud SDK
needs to be added for the offline workflow. Use `.venv-pi/bin/python` for
Gemini commands so the existing SDK remains available.

After approved OCR installation, these checks inspect local executable/language
data without recording images or playing audio:

```sh
tesseract --version
tesseract --list-langs
```

Confirm that `eng` appears in the language list. Native `say` is used for Mac
playback; Mac WAV export would require separately approved eSpeak NG setup.
The Pi's audio device, output routing, and comfortable volume still require a
controlled check with the user.

## Pi snapshot and offline OCR

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

## Offline speech or WAV export

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

Local Tesseract defaults are `ocr.language: eng`, `psm: 6`, `preprocess: grayscale`,
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
are `not_requested`, `pending`, `ready`, `no_text`, `failed`, or `cancelled`.
Gemini reading records a `pending` checkpoint before the request and includes
the configured model in `ocr.model`; local Tesseract has no cloud model.
Speech statuses are `not_requested`, `pending`, `skipped_ocr_failed`,
`skipped_cloud_failed`, `completed`, `failed`, or `cancelled`. The complete
OCR text is checkpointed before printing or starting
speech; report updates replace the previous JSON atomically. A speech failure
returns exit status 1 while preserving recognized text. Ctrl+C returns 130 and
attempts to record cancellation without discarding the preceding checkpoint.
Missing executable/language data, process failures, and timeouts must be
reported as such. Neither an empty OCR result nor any spoken text establishes
that navigation is safe.

## Controlled acceptance still to run

Start with the text-only Gemini command and a controlled printed page. Record
the command, saved source image, actual transcription, model, elapsed time,
and report. Check a successful empty scene separately from a cloud failure or
timeout: unavailable recognition must not be reported as “no readable text.”
No on-device Gemini OCR acceptance result is claimed yet.

After the corresponding local package approval and installation, record the
actual Pi command, requested/delivered image dimensions, printed ground truth, recognized text,
OCR latency, and artifact paths. Then check an empty scene, an image with more
than 60 words, camera/process failures, and interruption cleanup. Confirm
WAV synthesis separately from speaker/headphone audibility. Use the same
controlled page with network access unavailable to check the offline path.

Software tests exercise cloud responses, subprocesses, and hardware boundaries
with local fakes, including real harmless child-process timeout and
cancellation checks. Optional
real-engine tests generate English sign images for Tesseract and synthesize a
WAV with eSpeak NG without playback. They explicitly skip when the respective
engine is absent; no engine is downloaded by a test or application startup.
Cloud tests use mocked responses and do not establish on-device recognition
quality. No new on-device OCR or audio acceptance result is claimed here.
This prototype supplements established mobility aids and does not provide precise distance
measurements or guaranteed navigation safety.
