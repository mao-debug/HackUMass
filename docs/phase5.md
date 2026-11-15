# vidi Phase 5: requested voice commands

Phase 5 adds **M-triggered voice input** to the existing scene preview. Each
accepted M records one clip of at most five seconds, sends it to Gemini for
transcription, and passes the text through a local English command allowlist.
There is no continuous listening or wake-word service. Starting a profile,
loading a key, or opening the camera does not record or upload audio.

The user approved these short Gemini audio requests. The existing Google SDK,
saved key, and free-tier/no-billing-change scope are reused; no new Python
package download is needed. Google's [audio input guide](https://ai.google.dev/gemini-api/docs/audio)
documents inline WAV input, and the configured
[`gemini-3.5-flash-lite` model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
supports audio input. Transcription requires network access. Local proximity
and keyboard help remain available independently; keyboard D/T still need
Gemini for their image requests.

The Pi reports `/usr/bin/arecord` and an actual Bluetooth headset microphone
as its default input. The user recorded a requested five-second, 16 kHz mono
PCM16 WAV through `pulse`, played it with `paplay`, and confirmed hearing their
own recorded voice. A subsequent run logged HTTP 200, an accepted `describe`
voice command, camera scene text, and separate left/right sensor feedback.
The user confirmed hearing **both the scene description and printed-text
reading**. The supplied log does not show an accepted `read_text` voice
command; the text-reading result does not establish voice-triggered OCR.
The user has accepted Phase 4's functional camera/scene/sensor/audio checks
and deferred measured distance accuracy, acoustic crosstalk, wiring/supply,
sustained performance, and hardware upgrades to Phase 6. See
[hardware.md](hardware.md) and [Phase 4](phase4.md).

The observed camera image-age captions of 1.8 and 2.3 seconds refer to the
selected snapshots. They are not total microphone-to-answer latency or an
end-to-end performance measurement. A concise presentation sequence and
return-to-device checklist are in [demo.md](demo.md).

## Pi setup and first session

Use the existing `.venv-pi`, Google SDK, saved key, local eSpeak output, and
protocol-v1 ESP32 setup. Close other camera/serial programs first. Choose the
headset as the desktop **microphone input** as well as the playback output.
Read-only checks can identify the current source and installed capture tools:

```sh
cd ~/vidi
command -v arecord pactl
pactl get-default-source
pactl list short sources
arecord -L
```

The selected source must be the actual microphone, not an output monitor.
The default `pulse` ALSA device uses the existing desktop audio stack; the
application does not install an audio server or change input routing. Recheck
playback after selecting the Bluetooth microphone because available headset
profiles and sound quality can differ. Device addresses remain local; none
is embedded in the profiles or examples.

Start the full Pi session with the user's observed USB path:

```sh
python3 scripts/run_demo.py --target pi --cloud --voice --proximity --sensor-port /dev/ttyUSB0 --speak --run
```

The launcher reuses `.venv-pi/bin/python` and prints a unique report path under
ignored `artifacts/`. Without `--run`, it only prints a plan and opens nothing.
Cloud and voice options are explicit; the launcher reuses the installed
packages and current audio routing. The equivalent manual entry point remains:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-phase5.yaml --continuous --scene-live --preview --voice-commands --proximity --sensor-port /dev/ttyUSB0 --speak --report artifacts/phase5.json
```

If USB has renumbered, list ports as in [Phase 4](phase4.md) and use the actual
path. Omit `--proximity --sensor-port /dev/ttyUSB0` when testing without the
ESP32. Omit `--speak` to see results without playback. Neither omission enables
automatic recording. Keep the preview focused for keyboard input.

Press **M once**, wait for the recording status, and say one short command.
The default capture is five seconds of mono 16-bit PCM at 16 kHz, wrapped as
WAV in memory. Five seconds limits the clip, not the total response time.
Recording, transcription, a possible second image request, and speech each
take additional time.

## Keys and supported commands

`--voice-commands` requires `--scene-live --continuous --preview`. The new M,
T, and H controls are enabled in that session; existing D/P/S/R/q controls
remain available.

| Key | Action |
| --- | --- |
| M | Record one bounded clip, or read the explicitly selected replay WAV, then request Gemini transcription |
| D | Request one camera scene description |
| T | Request printed-text OCR from one current unannotated camera image |
| P | Request fresh local proximity; no audio or image upload |
| H | Show local command help; speak it only with `--speak` |
| S | Cancel local recording/speech and pending command/narration dispatch |
| R | Rearm failed voice/cloud actions and reconnect paused sensing; no upload by itself |
| q/Esc | Quit and cancel pending work |
| Ctrl+C | Cancel from the terminal |

The transcriber returns words; the local parser selects one allowlisted
action. Example accepted phrases are:

| Say after M | Action |
| --- | --- |
| “What's in front of me?” or “Describe the scene” | `describe` |
| “Read this sign” or “Read this text” | `read_text` |
| “Check proximity” or “What is nearby?” | `proximity` |
| “Stop speaking” | `stop` |
| “Help” or “What can you do?” | `help` |

Matching is against a complete supported phrase after case/punctuation
normalization. Unknown or unintelligible speech performs no requested camera
or sensor action. Model text is never executed as shell commands or Python.
Spoken “stop” requires an accepted M and cloud transcription; it is not an
always-listening interrupt. Use **S or q/Esc** for immediate local cancellation,
including when M is unavailable because another action is busy.

M itself uploads audio even when the recognized action is local proximity,
stop, or help. Use keyboard **P/H/S** to perform those local actions without
network access or an upload. D/T remain cloud image actions; they are not
offline alternatives to failed voice recognition.

## Staging, feedback, and failures

One voice-command chain is active at a time. Repeated M or conflicting busy
actions are ignored rather than queued. The shared cloud cooldown is at least
five seconds **after a completed cloud call**. A recognized describe/read
command can therefore wait after transcription before capturing and sending
its separate camera image. The waiting status is deliberate; it does not
indicate another recording. There are no automatic recording loops or API
retries. A failed stage pauses that path until explicit R and a new request.
The camera preview and independent local controls continue.

Descriptions and OCR refer to the selected image, shown separately from the
newest camera feed. Local proximity keeps independent left/right channels,
the provisional 1000 mm nearby threshold, and 0.5-second freshness policy.
It is not a measured distance to an object named in a description. Narration
uses a snapshot checked at speech start, not continuously updated readings
throughout the utterance. `--speak` enables bounded local playback; no Gemini
speech-generation API is called.

S or quitting cancels the recorder/local speech and suppresses later command
dispatch or narration. A cloud request already sent may finish; cancellation
does not recall an uploaded clip. Missing microphones, invalid WAV files,
unavailable services, and unknown speech produce explicit status without
inventing a command. Keyboard P/H/S/q remain local fallbacks. Existing OCR and
Gemini live detection commands retain their separate flags and profiles.

Audio bytes and the raw transcript are not written to reports or diagnostic
logs. Clips and transcription text stay transient in memory. Reports keep
the recognized action name, stage status, model/timing metadata, and requested
scene/OCR/proximity results. The complete returned OCR/scene text can still
appear in an explicitly requested report. Generated reports belong under
ignored `artifacts/`; do not place API keys in commands, YAML, or reports.

## Offline Mac rehearsal and optional WAV replay

Default planning and local rehearsal require no microphone, WAV, or cloud:

```sh
python3 scripts/run_demo.py
python3 scripts/run_demo.py --proximity --run
```

The first command only prints a plan. The second uses the synthetic profile
with the explicit offline guard: D/T/M are disabled, saved credentials are not
loaded, and H/P/S/R/Q remain local. P uses fixture data, not physical sensors;
optional `--speak` uses native Mac `say`. The launcher uses the existing
`.venv/bin/python` and generates a unique report path under ignored `artifacts/`.

The Mac profile uses synthetic pixels and invented proximity readings, marked
simulated. It does not include a recorded command asset. For M on Mac, supply
your own local **nonempty PCM16 WAV**, mono or stereo, 8–48 kHz and at most
five seconds. The bounded file limit is 1,000,000 bytes. The existing recorder
validates the file and sends an audio-only WAV representation when M is
accepted; it does not upload the file when starting the session.

```sh
python3 scripts/run_demo.py --cloud --voice --proximity --command-audio /path/to/your-command.wav --run
```

Replace the example WAV path before running. Optional `--speak` uses native
Mac `say`. Each accepted M replays that selected clip and requests Gemini
transcription; D/T also explicitly upload images. P/H/S remain local. Mac
simulation and replay are not physical Pi microphone acceptance.

Cloud features need `--cloud`; voice additionally needs `--voice`. Simulation
voice requires the replay WAV. `--python PATH` overrides the existing target
interpreter; `--artifacts-dir DIR` selects another unique-report directory.
`--prompt-key` requires cloud mode; no key value is passed to that flag.

## Configuration and evidence limits

| Setting | Default |
| --- | --- |
| `voice.device` | Literal ALSA device `pulse` |
| `voice.record_seconds` | 5; configurable integer 1–5 |
| `voice.timeout_s` | 30 seconds for cloud transcription |
| `voice.model` | null; uses existing `cloud.model`, currently `gemini-3.5-flash-lite` |
| `voice.audio_path` | null; explicit replay file instead of microphone capture |
| Recognition/OCR backends | `models.speech_recognition_backend: gemini`, `models.ocr_backend: gemini` |

`--microphone-device` overrides the literal ALSA input device; it requires
`--voice-commands` and cannot be combined with `--command-audio` or a configured
replay path. `--command-audio` also requires the voice flag. YAML paths resolve
relative to the YAML file; command-line paths resolve relative to the current
directory. Settings alone never begin capture.

The working Pi profile keeps 640×480 at a requested 30 FPS and the existing
settle/autofocus setup. Scene and image OCR request timeouts remain 60 seconds.
For a deliberately higher-resolution printed-text snapshot, the separate
1280×960 [Phase 3 Gemini OCR profile](phase3.md) remains available; stop this
live session before opening that camera command.

Local microphone capture/playback and voice-to-`describe` action routing now
have user-run Pi evidence. The user also heard printed-text output, but a
voice-to-`read_text` trigger is not established by the supplied log. These
functional results do not establish recognition accuracy, every allowlisted
command, complete transcription accuracy, or total response latency. The
previously deferred measured distance/crosstalk,
wiring/supply, sustained FPS, wearable mounting, and hardware upgrades remain
Phase 6 work at the user's request. The prototype supplements established
mobility aids and provides no navigation-safety guarantee.
