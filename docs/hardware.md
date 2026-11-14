# vidi hardware inventory

Record exact models, quantities, connections, and observations as equipment is
verified. Availability reflects the user's inventory and does not certify that
equipment has passed an integration test.

| Item | Model or requirement | Status | Verification notes |
| --- | --- | --- | --- |
| Current computer | Raspberry Pi 4 | Available | RAM, board revision, OS release/architecture, Python, and quantity to inspect |
| Existing desktop access | Partner's monitor and keyboard | Connected to the Pi | Preserve the partner's OS installation; use these to inspect it first |
| Requested computer | Raspberry Pi 5, 8 GB RAM | Requested; unverified | Delivery and compatible power supply to verify |
| Camera | Raspberry Pi AI Camera (IMX500) | Available | Ribbon cable, connection, capture, inference, and model labels remain unverified |
| Storage | MicroSD card | To inspect | Preserve installed OS and files; record capacity, image, and available space |
| Portable power | Power bank | To verify | Output capability, runtime, cable, and board compatibility to verify |
| Voice input | Microphone | To source or verify | Deferred from Phase 2; interface, quality, and placement unverified |
| Speech output | Headphones or speaker | To source or verify | Deferred from Phase 2; interface and volume control unverified |
| Wearable mounting | Glasses frame and camera accessories | To verify | Fit, attachment, field of view, and cable strain relief to check |
| Activation input | Optional pushbutton | Optional; unverified | Wiring and software integration deferred |

Phase 2 targets local detection on the Pi 4 using the AI Camera and an official
SSD MobileNet FPNLite 320 packaged model. macOS uses synthetic/image input and
explicitly simulated detection fixtures. Pi 5 operation, wearable integration,
audio, OCR, and voice commands have not been verified by these software tests.

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

Use the inspection, install policy, and acceptance procedure in
[phase2.md](phase2.md). Approve any system package changes, model/label downloads,
SDK installation, or live cloud uploads separately. If the existing OS is
incompatible, stop and propose a supported image on a separately approved spare
microSD card rather than overwriting the current installation.

## Bench record

No physical acceptance result is claimed. Fill in this record on the Pi:

| Field | Observed value |
| --- | --- |
| Test date, operator, board revision/RAM | Not measured |
| OS identity, architecture, Python executable/version | Not inspected |
| NumPy/OpenCV/Picamera2/libcamera versions and module origins | Not inspected |
| Model path, model/package version, labels source | Not verified |
| Camera startup time and requested/actual frame size | Not measured |
| 100-frame offline report and observed processing rate | Not run |
| Missing model, unplugged camera, and interruption cleanup | Not tested |
| Optional one-shot cloud test, project tier, terms review | Not authorized or run |

Use controlled bench scenes before wearable trials. The prototype supplements
established mobility aids and cannot guarantee obstacle coverage, precise
distance, or navigation safety.
