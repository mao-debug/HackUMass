# vidi Phase 6 deployment and performance plan

Phase 6 will measure and improve the working prototype, then prepare a
repeatable demonstration. The current baseline is a Raspberry Pi 5 with an
IMX708 camera, two HC-SR04 sensors connected through an ESP32-WROOM-32UE USB
board, requested Gemini image and voice processing, and local eSpeak playback
through Bluetooth headphones. The user has heard a recognized voice command
produce a scene description and has separately confirmed spoken text reading.
The trigger for that text reading was not established, so voice-triggered OCR
acceptance remains unconfirmed.

These are functional passes. Sensor distance accuracy, acoustic interference,
sustained performance, power, and wearable operation remain unverified. The
user explicitly deferred those measurements and upgrades to this phase.

## Preserve the working baseline

From the Mac, review and checkpoint the working feature branch, configuration,
firmware source, and setup instructions before changing a backend or tuning
parameters. Keep changes on feature branches and record the tested commit.
Keep existing keys, generated recordings, reports, and models out of Git.

When the Pi is available, record its installed package versions, selected
audio input/output, camera profile, ESP32 firmware identity, and actual USB
path. Make a full SD-card backup using a verified source card and destination;
power the Pi down before removing the card. Keep that image private because
it includes the saved credentials. Restore-test on a spare card if available.
This is a planned manual backup, not a backup already performed.

## Measure the sensor assembly first

Confirm the actual USB carrier, flash capacity, and board settings before any
new firmware upload. The WROOM-32UE module marking alone does not establish
the carrier or installed flash capacity. Inspect both HC-SR04 echo paths:
each 5 V Echo needs its own verified voltage protection and a common ground.
Record resistor values and measured echo voltage against the ESP32 input
limits, following the [firmware wiring notes](../esp32code/README.md). Change
wiring only with power removed.

Test a single channel in isolation, then both channels operating together.
Use a large flat stationary target facing the tested sensor directly, with
the reference distance measured from the transducer face. Keep hands, cables,
and mounting parts out of its view. Collect **50 attempts per channel at
0.5, 0.8, 1.2, and 1.5 m**, preserving the complete JSONL output and reference
distance for each run. The earlier moving, estimated-distance excerpts are
functional checks and cannot supply this accuracy evidence.

For every position, calculate valid returns out of all attempts, median
absolute error among valid returns, and the counts of missing/invalid returns.
Compare isolated and paired operation to assess interference. The initial
bench targets are **at least 45 valid returns out of 50** and **median absolute
error at most 100 mm**; these are provisional criteria, not achieved results.
Also move a target closer suddenly, remove it, and introduce a second target.
Record incorrect channel attribution and jumps. Keep left/right readings
independent; two returns do not establish a centered object or its identity.

## Measure responsiveness before tuning

Use the same camera settings, lighting, power source, and desktop session for
three short comparisons: camera only; camera plus proximity; and the full
requested voice/image/speech workflow. Start with 60-second runs, then perform
a 15-minute full-workflow run. Record requested FPS separately from measured
capture rate and observed preview responsiveness. The provisional goal is to
retain **at least 80% of the camera-only capture rate** without growing queues
or blocking controls. Earlier Pi measurements are context, not this matched
comparison.

The host proximity monitor caps polling by waiting at least 200 ms after a
completed pair before starting another. Source setup and thread scheduling
cannot consume that gap. Actual paired throughput is therefore below 5 Hz
when acquisition takes time; measure it separately from camera FPS before
changing the schedule.

Measure the stages of several accepted commands separately: local recording,
transcription, cooldown wait, camera request, and speech. Default recording
takes five seconds; the shared cloud cooldown adds at least five seconds
after transcription before a recognized scene/read request can start. Five
seconds is therefore not an end-to-end response-time promise. Record both
time to first audible feedback and time until the utterance finishes.

The Phase 5 report already provides `startup_s`, `processing_s`, `capture_fps`,
`voice.elapsed_s`, `description.elapsed_s`, `ocr.elapsed_s`, and
`speech.elapsed_s`. Its bounded `actions` records contain `monotonic_ns`
timestamps and stage/status names. Use those to reconstruct recording and
dispatch intervals where the required events are present; a service-stage
elapsed value alone does not include the complete command chain. Reports
retain only the latest 120 action records, so use short measured runs or
checkpoint separate reports. Add explicit timing events if a required stage
cannot be reconstructed; do not substitute an unrelated duration.

Check the selected frame ID and capture timestamp, plus the age announced at
speech start. A fast camera preview does not make a delayed description
current. Keep raw audio and raw transcripts out of performance reports.

## Verify power and mounting under load

Record the actual Pi supply and cable, cooling arrangement, temperatures, and
available voltage/throttling indicators during the sustained run. Repeat on
the intended portable power bank and measure runtime rather than estimating
it from advertised capacity. Investigate throttling, resets, camera drops,
USB disconnects, or audio interruptions before increasing workload.

Fit the camera and sensors to the glasses only after the bench checks. Verify
mount orientation, sensor clearance, balance, strain relief, CSI/USB cable
routing, connector retention, and access to controls. Repeat the sensor and
camera checks in the final mounting because placement changes the tested
geometry. This project does not include a cane. Hardware activation buttons
are an optional next step: confirm the actual button/wiring and implement
debouncing and explicit bounded actions before replacing keyboard controls.

## Test failures and select upgrades from evidence

Recheck USB removal/reconnection and a missing individual sensor. Missing,
invalid, or stale readings must become unavailable, never clearance. Measure
expiry against the configured 0.5-second freshness policy. Check microphone
loss, Bluetooth reconnection, silence, unknown/negated commands, busy-key
suppression, and local S/q/Ctrl+C cancellation during recording, a request,
cooldown, and speech. An already submitted request may finish; cancelled work
must not dispatch a later image request or start late narration.

With the network unavailable, verify keyboard P/H/S and cleanup still work.
Voice recognition and D/T should fail clearly and wait for explicit recovery;
they must not silently retry or substitute a different service. Confirm the
camera and serial port can be reopened after quitting.

Choose upgrades one at a time and repeat the affected comparison:

- For clearer speech, audition the optional Piper voice against eSpeak with
  the same phrases and headphones, measuring intelligibility and delay.
  Piper runtime and voice-model downloads still require approval of exact
  packages, model, and sources; the existing backend remains the baseline.
- For offline reading, obtain approval for the exact Tesseract engine and
  language-data installation, then test known printed text without networking.
  Existing cloud OCR does not establish offline OCR acceptance.
- For offline detection, evaluate a replacement for the temporary, inaccurate
  MobileNet-SSD baseline against labelled scenes and measured Pi performance.
  Model downloads or backend/service changes require separate approval.
- Use latency and failure measurements to justify resolution, focus, request
  scheduling, or model changes. Preserve explicit uploads and bounded speech.

Request approval before any new package, model, service, or paid resource,
stating its exact source and intended use. No budget increase or service change
is implied by this plan.

## Demonstration acceptance record

| Workflow | Current evidence | Phase 6 evidence to record |
| --- | --- | --- |
| Requested scene and local speech | User heard output with a smooth Pi preview | Labelled scene checks, latency, selected-frame age, matched FPS |
| Requested voice commands and text reading | User heard voice-triggered description and separate read-aloud; voice-triggered OCR not confirmed | Verify spoken read command, repeated command success, unknown/negated input, per-stage timing |
| Two-channel proximity | Valid protocol samples and changing returns | Stationary measured-distance results and interference comparison |
| USB failure and recovery | Logs show both unavailable, then readings resumed | Timed expiry/recovery and audible unavailable feedback |
| Local fallback and cancellation | Keyboard controls implemented | Network/input-loss trials, no late actions, successful restart |
| Portable wearable operation | Not validated | Mounting checks, sustained power/thermal run, measured battery runtime |

Prepare a short repeatable demo using verified commands and a printed test
sign, with local fallback controls available. Save the tested commit,
configuration, actual measurements, and remaining limitations together.
Present vidi as an assistive prototype that complements established mobility
aids; the demonstration must not claim precise obstacle localization or
guaranteed navigation safety.
