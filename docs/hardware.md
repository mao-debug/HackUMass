# vidi hardware inventory

Record exact models, quantities, connections, and observations as equipment is
verified. Availability reflects the user's inventory and does not certify that
equipment has passed an integration test.

| Item | Model or requirement | Status | Verification notes |
| --- | --- | --- | --- |
| Current computer | Raspberry Pi 4 | Available | Debian 12, aarch64, Python 3.11.2 confirmed; RAM/board revision to record |
| Existing desktop access | Partner's monitor and keyboard | Connected to the Pi | Use direct desktop/GitHub or USB handoff; preserve the existing installation |
| Requested computer | Raspberry Pi 5, 8 GB RAM | Requested; unverified | Delivery and compatible power supply to verify |
| Connected camera | Camera Module 3, IMX708 | Capture verified in supplied Pi logs | Two runs of 3 frames at 640×480; image appearance and CPU detection still to verify |
| Optional AI Camera | IMX500 | Not confirmed as connected | Earlier planned backend remains optional; verify a separate sensor before using its profile |
| Storage | Installed microSD card | In use | Preserve Debian installation/files; capacity and available space to record |
| Portable power | Power bank | To verify | Output capability, runtime, cable, and board compatibility to verify |
| Voice input | Microphone | To source or verify | Deferred from Phase 2; interface, quality, and placement unverified |
| Speech output | Headphones or speaker | To source or verify | Deferred from Phase 2; interface and volume control unverified |
| Wearable mounting | Glasses frame and camera accessories | To verify | Fit, attachment, field of view, and cable strain relief to check |
| Activation input | Optional pushbutton | Optional; unverified | Wiring and software integration deferred |

The connected sensor supersedes the earlier AI Camera assumption. Phase 2 now
targets Picamera2 capture plus OpenCV DNN CPU inference with the approved
MobileNet-SSD Caffe model. The OS OpenCV 4.6.0 is installed alongside NumPy
1.24.2, PyYAML 6, setuptools 66, and Picamera2 0.3.27. No additional inference
package is required. See [models.md](models.md) for approved fixed-commit model
downloads, hashes, and license. macOS fixture replay remains explicitly
simulated; a real-model synthetic-frame smoke test checks setup only.

Run the read-only prerequisite report before making changes:

```sh
python3 scripts/check_hardware.py
python3 scripts/check_hardware.py --json
```

It uses only Python's standard library. It reports local OS/Python identity,
discoverable module origins, distribution-metadata versions, command paths, and
possible camera device paths. It does not import optional hardware packages,
open devices, record audio, install software, or contact the network. An
apt-provided module can lack PyPI distribution metadata; neither its presence
nor its path proves ABI or model compatibility. Camera, microphone, and audio
operation remain **unverified** in this report.

Use the setup and acceptance procedure in [phase2.md](phase2.md). Preserve the
working OS package stack; further package changes, unrelated downloads, and
live cloud uploads require approval. The CPU model download is approved, and
the optional Google SDK was installed and tested locally on Mac. If the existing OS is
incompatible, stop and propose a supported image on a separately approved spare
microSD card rather than overwriting the current installation.

## Bench record

The supplied Pi terminal output confirms camera capture at commit `827577d`.
Full detection acceptance remains pending. Continue filling in this record:

| Field | Observed value |
| --- | --- |
| Test date, operator, board revision/RAM | Not measured |
| Capture evidence received | 2026-11-14; terminal output from `simaofea@TheDon`, commit `827577d` |
| OS identity, architecture, Python executable/version | Debian 12, aarch64, Python 3.11.2 reported on Pi |
| NumPy/OpenCV/Picamera2/PyYAML/setuptools | 1.24.2 / 4.6.0 / 0.3.27 / 6 / 66 reported; import origins to record |
| Camera enumeration | rpicam/libcamera report IMX708 |
| vidi camera-only capture | Passed twice: 3 frames per run at 640×480; `source=picamera2`, `detector=none`, no reported errors |
| Image export and visual inspection | Saved to `/home/simaofea/vidi/artifacts/pi-camera.png`; image contents, orientation, and colors not yet inspected |
| CPU model assets | Fixed-commit author files downloaded/tested locally on Mac; transfer or approved download to Pi still required |
| CPU model load/forward | Passed on Mac synthetic frame with zero detections; not a Pi or accuracy test |
| Camera startup time and requested/actual frame size | Startup duration not measured; requested and delivered 640×480 |
| 100-frame offline report and observed processing rate | Not run |
| Missing model, unplugged camera, and interruption cleanup | Not tested |
| Optional one-shot cloud test, project tier, terms review | Not authorized or run |

Use controlled bench scenes before wearable trials. The prototype supplements
established mobility aids and cannot guarantee obstacle coverage, precise
distance, or navigation safety.
