# vidi hardware inventory

Record exact models, quantities, connections, and observations as equipment is
verified. Availability reflects the user's inventory and does not certify that
equipment has passed an integration test.

| Item | Model or requirement | Status | Verification notes |
| --- | --- | --- | --- |
| Current computer | Raspberry Pi 5 | User reports using the new board with the Pi 4 SD card | Exact board identity, RAM, boot/runtime, power, and cooling checks pending; the initially requested 8 GB is unverified |
| Previous computer | Raspberry Pi 4 | Available; earlier validation platform | Debian 12/Bookworm, kernel 6.12, aarch64, Python 3.11.2 reported; RAM/board revision not recorded |
| Existing desktop access | Partner's monitor and keyboard | Previously connected to the Pi 4 | Verify the Pi 5 desktop connection; use direct desktop/GitHub or USB handoff and preserve the existing installation |
| Connected camera | Camera Module 3, IMX708 | Capture and CPU inference verified on Pi 4; Pi 5 pending | Pi 4: 20 live frames at 640×480, all inference-ready; user confirmed a person box in a follow-up controlled test |
| Pi 5 camera ribbon | Standard–Mini 15-to-22-pin cable | To verify | Pi 4's 15-to-15-pin ribbon does not fit Pi 5 directly; disconnect power before changing it |
| Optional AI Camera | IMX500 | Not confirmed as connected | Earlier planned backend remains optional; verify a separate sensor before using its profile |
| Storage | Existing Pi 4 microSD card | User reports moving it to Pi 5 | Preserve the OS, `.venv-pi`, saved key, models, and artifacts; capacity/free space and Pi 5 boot to record |
| Portable power | Power bank | To verify | Output capability, runtime, cable, and board compatibility to verify |
| Pi 5 bench power/cooling | Suitable USB-C supply/cable and cooling | To verify | Record supply profile, fan/heatsink, temperatures, warnings, and throttle flags before sustained work |
| Voice input | Microphone | To source or verify | Deferred from Phase 2; interface, quality, and placement unverified |
| Speech output | Bluetooth headphones requested | Prior audible speech reported; Pi 5 routing/playback pending | User reported the earlier voice was working but robotic; verify pairing, selected output, and audible playback on the new board |
| Wearable mounting | Glasses frame and camera accessories | To verify | Fit, attachment, field of view, and cable strain relief to check |
| Activation input | Optional pushbutton | Optional; unverified | Wiring and software integration deferred |

The confirmed Pi 4 sensor superseded the earlier AI Camera assumption. Local
detection uses Picamera2 capture plus OpenCV DNN CPU inference with the approved
MobileNet-SSD Caffe model. The retained Pi 4 installation has OS OpenCV 4.6.0
alongside NumPy 1.24.2, PyYAML 6, setuptools 66, and Picamera2 0.3.27; this
package stack has not yet been checked on Pi 5. No additional CPU inference
package is required. See [models.md](models.md) for approved fixed-commit model
downloads, hashes, and license. macOS fixture replay remains explicitly
simulated; a real-model synthetic-frame smoke test checks setup only.

Follow [pi5-migration.md](pi5-migration.md) for the current SD-card migration:
inspect the retained installation, verify the 15-to-22-pin camera connection,
enumerate the sensor, and run camera-only preview before inference. Every
physical Pi 5 acceptance result remains pending.

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

The user reports moving the same SD card into a new Pi 5. No Pi 5 diagnostic,
camera, power/thermal, inference, or audible-speech acceptance result is
recorded yet. Record new evidence using the
[migration checklist](pi5-migration.md#pi-5-acceptance-record); keep the Pi 4
timings above identified with their original board.

Use controlled bench scenes before wearable trials. The prototype supplements
established mobility aids and cannot guarantee obstacle coverage, precise
distance, or navigation safety.
