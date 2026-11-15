# vidi hardware inventory

Record exact models, quantities, connections, and observations as equipment is
verified. Availability reflects the user's inventory and does not certify that
equipment has passed an integration test.

| Item | Model or requirement | Status | Verification notes |
| --- | --- | --- | --- |
| Current computer | Raspberry Pi 5 Model B Rev 1.1 | Board/boot/runtime identity confirmed in supplied report | Kernel sees 8061 MiB, consistent with an 8 GB board; exact physical RAM capacity not independently verified; further power/cooling checks deferred to Phase 6 |
| Previous computer | Raspberry Pi 4 | Available; earlier validation platform | Debian 12/Bookworm, kernel 6.12, aarch64, Python 3.11.2 reported; RAM/board revision not recorded |
| Existing desktop access | Partner's monitor and keyboard | Previously connected to the Pi 4 | Verify the Pi 5 desktop connection; use direct desktop/GitHub or USB handoff and preserve the existing installation |
| Connected camera | Camera Module 3, IMX708 | Pi 5 enumeration and camera-only preview passed | Index 0; 282 frames at 640×480, measured 21.79 FPS; further measured image-quality checks deferred to Phase 6 |
| Pi 5 camera ribbon | Standard–Mini 15-to-22-pin cable | Sensor detected; exact cable/strain-relief inspection not recorded | Pi 4's 15-to-15-pin ribbon does not fit Pi 5 directly; disconnect power before changing it |
| Optional AI Camera | IMX500 | Not confirmed as connected | Earlier planned backend remains optional; verify a separate sensor before using its profile |
| Storage | Existing Pi 4 microSD card | Moved to Pi 5; boot identity reported | Preserve the OS, `.venv-pi`, saved key, models, and artifacts; capacity/free space to record |
| Portable power | Power bank | To verify | Output capability, runtime, cable, and board compatibility to verify |
| Pi 5 bench power/cooling | Suitable USB-C supply/cable and cooling | Measured checks deferred to Phase 6 | Supply profile/quality, fan/heatsink, temperatures, warnings, and throttle flags not fully recorded |
| Voice input | Bluetooth headset microphone | Pi local capture/playback passed by user confirmation | Actual Bluetooth input selected; requested five-second 16 kHz mono PCM16 capture via `arecord --device pulse` and `paplay` was audible; command recognition/routing still pending |
| Speech output | Bluetooth headphones | Pi 5 test word and Gemini OCR read-aloud passed by user confirmation | User heard `espeak-ng vidi` and the subsequent printed-text reading; optional neural voice remains pending |
| Proximity controller | ESP32-WROOM-32UE module on a USB development board with CP2102N bridge | `/dev/ttyUSB0` and real protocol-v1 paired samples supplied | Sequences 0–2 had both channels `ok`; exact carrier revision, flash capacity, and separate upload log not recorded |
| Proximity channels | Two HC-SR04 ultrasonic sensors | Functional readings/movement and combined Pi use confirmed | Left TRIG/ECHO 25/34, right 26/35; measured accuracy, crosstalk, divider/echo voltage, and supply checks deferred to Phase 6 |
| Wearable mounting | Glasses frame and camera accessories | To verify | Fit, attachment, field of view, and cable strain relief to check |
| Activation input | Optional pushbutton | Optional; unverified | Wiring and software integration deferred |

The IMX708 sensor supersedes the earlier AI Camera assumption. Local detection
uses Picamera2 capture plus OpenCV DNN CPU inference with the approved
MobileNet-SSD Caffe model. The Pi 5 report finds NumPy 1.24.2, PyYAML 6,
setuptools 66.1.1, Google SDK 2.34.0, and Picamera2 0.3.31; the earlier Pi 4
record used Picamera2 0.3.27. OS-provided `cv2` and `libcamera` are discoverable.
The subsequent Pi 5 application run demonstrated usable native camera capture,
OpenCV/NumPy processing, and preview; the exact imported OpenCV version is still
unreported. Missing `opencv-python` distribution metadata does not mean apt's
`cv2` is absent. No new installation or update is indicated by this evidence.
See [models.md](models.md) for approved fixed-commit model
assets, hashes, and license. macOS fixture replay remains simulated; a
real-model synthetic-frame smoke test checks setup only.

Follow [pi5-migration.md](pi5-migration.md) for the current SD-card migration:
inspect the retained installation, verify the 15-to-22-pin camera connection,
enumerate the sensor, and run camera-only preview before inference. Pi 5
identity, enumeration, and camera-only preview are confirmed. Measured preview
rate was 21.79 FPS in the camera-only check. A subsequent Gemini live run
captured 684 frames at 23.43 FPS with four successful responses and no
unavailable results. The user also confirmed hearing the Pi 5 eSpeak test word
and a subsequent Gemini OCR reading, which reported `OCR=ready` and
`speech=completed`. Further measured image quality, recognition accuracy,
power/thermal behavior, and local-model hardware checks are deferred to
Phase 6 at the user's request. The optional neural voice remains unverified
and subject to its pending download approval.

Run the read-only prerequisite report before making changes:

```sh
python3 scripts/check_hardware.py
python3 scripts/check_hardware.py --json
```

It uses only Python's standard library. It reports local board/OS/Python identity,
discoverable module origins, distribution-metadata versions, command paths, and
possible camera device paths. It does not import optional hardware packages,
open devices, record audio, install software, or contact the network. An
apt-provided module can lack PyPI distribution metadata; neither its presence
nor its path proves ABI or model compatibility. Camera, microphone, and audio
operation remain **unverified** in this report.

Use the setup and acceptance procedure in [phase2.md](phase2.md). Preserve the
working OS package stack; further package changes and unrelated downloads
require specific approval. The CPU model and Pi Google SDK were approved, and
explicit Gemini snapshot/live image uploads are approved under the existing
free-tier, no-billing-change scope. A saved key never activates uploads alone.
Pi eSpeak NG installation is approved; optional Piper downloads remain pending.
Phase 4 pyserial 3.5 and Arduino-ESP32 core 3.3.12/compiler downloads are approved;
pyserial is installed on Mac, and a local real ESP32 cross-compile passed.
Subsequent Pi protocol samples and combined camera/scene/sensor/audio operation
were supplied; a separate upload log and measured hardware bench remain
unrecorded. The user accepted current functional testing and explicitly
deferred further hardware testing/upgrades to Phase 6 while development
continues. This does not certify all hardware or measured distance accuracy.
Follow [phase4.md](phase4.md) and the
[firmware instructions](../esp32code/README.md) before opening the USB port.
If the existing OS is
incompatible, stop and propose a supported image on a separately approved spare
microSD card rather than overwriting the current installation.

## Pi 4 bench record

The supplied Pi 4 terminal output confirms camera capture at commit `827577d`
and a later 20-frame live CPU detection run. The user confirmed that a follow-up
person test placed the box over the person's body. Broader recognition checks
and longer offline acceptance remain pending. Continue filling in this record:

| Field | Observed value |
| --- | --- |
| Test date, operator, board revision/RAM | Not measured |
| Capture evidence received | 2026-11-14; terminal output from `simaofea@TheDon`, commit `827577d` |
| OS identity, architecture, Python executable/version | Debian 12, aarch64, Python 3.11.2 reported on Pi |
| NumPy/OpenCV/Picamera2/PyYAML/setuptools | 1.24.2 / 4.6.0 / 0.3.27 / 6 / 66 reported; import origins to record |
| Camera enumeration | rpicam/libcamera report IMX708 |
| vidi camera-only capture | Passed twice: 3 frames per run at 640×480; `source=picamera2`, `detector=none`, no reported errors |
| Image export and visual inspection | Saved to `/home/simaofea/vidi/artifacts/pi-camera.png`; image contents, orientation, and colors not yet inspected |
| CPU model assets | Approved downloads completed on Pi; checksum verification output not supplied |
| CPU model load/forward | Passed on Pi saved camera image: 1 ready frame, zero detections; Mac synthetic smoke also passed |
| Live camera and CPU inference | 20 frames, 20 ready, 0 not ready; backend `opencv_ssd`, source `picamera2`, simulated `false` |
| Initial live final-frame prediction | Class 15 `person`, model score 0.653709; box `[517, 192, 640, 466]`, at the image's right edge; user identified the visible subject as a hand, so this was a partial-body case |
| Follow-up person recognition check | User confirmed the person box covered the body after requesting a subject with head and torso visible; follow-up JSON, score, and timing not supplied |
| Camera startup time and requested/actual frame size | Reported camera-worker startup 3.386 s; requested and delivered 640×480 |
| 20-frame processing timing | Processing loop 13.260 s, approximately 1.51 frames/s including initial frame wait; camera startup, model loading, teardown, and export excluded |
| Reported frame-interval rates | `capture_fps` and `inference_ready_fps` both 2.00015; based on capture timestamps, not isolated model inference timing |
| Live output artifacts | `/home/simaofea/vidi/artifacts/pi-live-detections.png` and `pi-live.json` saved |
| Image viewer | Images opened according to the user; `eom` emitted missing Peas/PeasGtk typelib warnings after vidi completed; no package changes made |
| 100-frame offline report and observed processing rate | Not run |
| Missing model, unplugged camera, and interruption cleanup | Not tested |
| Gemini snapshot detection | User reported a successful Pi snapshot run; broader controlled recognition checks pending |
| Gemini printed-text reading | Supplied log shows HTTP 200 and OCR `ready`, printing an MLH Hack Days flyer; `artifacts/pi-gemini-reading.json` saved; speech was `not_requested` |
| Gemini live preview | Supplied run reported approximately 11.21 preview FPS; its Gemini request timed out and uploads paused, so this was not a successful empty detection |
| Speech/headphone check | User subsequently reported working audible speech and a robotic voice; exact output-device details and a controlled complete read-aloud record were not supplied |

The model score is not measured recognition accuracy. The reported interval
rates use `(frame_count - 1) / (last_capture_time - first_capture_time)`;
they omit the initial wait for a frame and the final frame's inference work.
Processing throughput uses `frame_count / processing_s` instead. The requested
2 FPS run limit also means this is not a maximum-speed detector benchmark.

## Pi 5 migration record

The supplied Pi 5 reports confirm identity/discovery and a subsequent working
camera-only application run. Discovery alone does not certify native-library
compatibility; the run provides separate evidence for the exercised capture,
OpenCV/NumPy processing, and preview path.

| Field | Observed value |
| --- | --- |
| Board model | Raspberry Pi 5 Model B Rev 1.1 |
| OS and kernel | Debian 12/Bookworm; `6.12.115+rpt-rpi-2712` |
| Architecture/process bitness | aarch64 / 64-bit |
| Kernel-visible memory | 8061 MiB, consistent with an 8 GB board; physical capacity not independently established |
| Python runtime | 3.11.2; `/home/simaofea/vidi/.venv-pi/bin/python`; virtual environment true |
| Distribution metadata | NumPy 1.24.2; PyYAML 6; setuptools 66.1.1; Google SDK 2.34.0; Picamera2 0.3.31 |
| Native module discovery | OS `cv2` and `libcamera` present; `opencv-python` metadata null is expected for apt-provided OpenCV and is not an absent-module result |
| Camera enumeration | Index 0, IMX708, 4608×2592; `/base/axi/pcie@1000120000/rp1/i2c@80000/imx708@1a` |
| Camera software/pipeline | libcamera `0.5.2+99-bfd68f78`; libpisp v1.2.1, hash `981977ff21f3`; tuning `rpi/pisp/imx708.json`; PiSP variant `BCM2712_D0`, CFE `/dev/media3`, ISP `/dev/media1` |
| Configured streams | Main 640×480 RGB888; raw 1536×864 PISP_COMP1 |
| Camera-only continuous run | 282 captured frames, measured 21.79 FPS versus requested 30 FPS; detector `none`, 0 successful analyses and 0 unavailable results, as expected with no detector |
| Preview completion | User returned to the normal terminal prompt after q; no traceback or fatal error reported; colors/clarity not yet assessed |
| Requested exports | `artifacts/pi5-camera.png` and `artifacts/pi5-camera.json`; no export error reported, but file contents were not independently read or viewed |
| Nonfatal warnings in this run | Repeated libpng `iCCP: known incorrect sRGB profile` and one GLib-GObject `g_object_unref` critical; run finished; warning origins unconfirmed and no driver reinstall justified by these messages alone |
| Command discovery | eSpeak NG, rpicam, vcgencmd, bluetoothctl, wpctl, pactl, paplay, pw-play, and aplay available |
| Optional dependencies | Piper absent in camera environment, downloads pending; local Tesseract and optional microphone modules absent; voice capture uses existing `arecord` instead of those Python modules |
| Earlier environment-path error | `.venv-pi/bin/python` was run from `~`; resolved after `cd ~/vidi`; environment rebuild not needed |
| Local speech | User confirmed hearing `espeak-ng vidi` on Pi 5 after the headphone/output check |
| Gemini live identification | 684 camera frames, measured 23.43 FPS, 4 successful analyses, 0 unavailable results; four HTTP 200 replies |
| Gemini returned predictions | Frame 1: ceiling/light fixture/vent/light; frame 229: person/light; frame 377: marker/person; frame 569: cell phone/person. These are model outputs, not independently checked ground truth. |
| Gemini live warnings/completion | Repeated libpng iCCP warnings remained nonfatal; run completed normally with the final capture summary |
| Gemini OCR and local speech | HTTP 200, `OCR=ready`, `speech=completed`; user confirmed hearing the spoken excerpt; report saved to `artifacts/pi5-gemini-speaking.json` |
| OCR transcription observations | MLH Hack Days flyer text was returned, with partial lines near the bottom; complete transcription accuracy was not checked against the source |
| Phase 4 ESP32/USB evidence | ESP32-WROOM-32UE module and CP2102N bridge at `/dev/ttyUSB0`; supplied protocol-v1 paired samples seq 0–2 were real (`simulated: false`) with both channels `ok`; photos also show board/connection and pin labels; carrier revision, flash capacity, separate upload log, resistor measurements, and echo voltage not recorded |
| Phase 4 native software verification | Six local tests passed, compiling the actual sketch with temporary SDK stubs; these do not certify actual ISR timing, wiring, or acoustics |
| Phase 4 actual target compile | Clean rebuild passed using Arduino CLI 1.6.0 and Espressif core 3.3.12; FQBN `esp32:esp32:esp32:FlashMode=dio,FlashFreq=40,FlashSize=4M,PSRAM=disabled,UploadSpeed=115200`, all warnings enabled, no warnings; 276476 program bytes (21%), 22788 static RAM bytes (6%); no device opened or flashed |
| Local build artifacts | Ignored tools/config `artifacts/tools/arduino/`, build tree `artifacts/firmware-build/`, binaries `artifacts/firmware/` including `esp32code.ino.bin`; generated files not committed |
| Phase 4 combined operation | User confirmed camera, requested Gemini descriptions, local sensor feedback, and audible speech worked together with a smooth preview; sustained FPS comparison not measured |
| Phase 5 local microphone | Actual Bluetooth input confirmed as default; user recorded a requested five-second WAV through ALSA `pulse` (16 kHz, mono, S16_LE) to `artifacts/mic-check.wav`, played it with `paplay`, and explicitly heard their recorded voice; no personal device address recorded |
| Phase 5 Gemini voice commands | Physical audio transcription and allowlisted action routing remain pending; local recording/playback is a separate functional pass |
| Phase 4 USB disconnect/recovery | Subsequent live logs showed both channels unavailable on USB disconnect and readings resumed afterward; audible disconnect feedback was not separately confirmed |
| Phase 4 movement excerpts | Two excerpts of 15 paired records each, seq 35–49 and seq 28–42; user said the setup moved and distances were estimated by eye; these are functional movement checks |
| 50 cm accuracy trial | Not validated: no confirmed stationary measured reference or complete 50-record accuracy result; the visible readings are not an accuracy pass |
| Deferred sensor bench | Phase 6, at the user's explicit request: one sensor then both, 50 attempts at 0.5/0.8/1.2/1.5 m; 90% valid and median absolute error ≤10 cm remain provisional targets |
| Deferred Phase 4 measurements | Phase 6: measured accuracy/crosstalk, exact wiring/divider/echo voltage and supply quality, sustained preview FPS comparison, timed fault/cleanup behavior, and remaining recognition-quality checks |
| Pending acceptance | Exact OpenCV version, image colors/clarity and exported-file inspection, recognition accuracy, local inference, supply/temperature/throttle checks, and optional neural voice |

Retain the [migration checklist](pi5-migration.md#pi-5-acceptance-record) for
the deferred Phase 6 work; the user has chosen to continue software development
with the functional evidence above. Keep all Pi 4 measurements attributed to
their original board. No package installation or update is required by the
supplied report so far.

Use controlled bench scenes before wearable trials. The prototype supplements
established mobility aids and cannot guarantee obstacle coverage, precise
distance, or navigation safety.
