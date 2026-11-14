# VisionBridge hardware inventory

Record exact models, quantities, connections, and test results as equipment is
verified. The statuses below reflect the project information supplied so far;
they do not certify that any hardware has passed an integration test.

| Item | Model or requirement | Status | Verification notes |
| --- | --- | --- | --- |
| Current computer | Raspberry Pi 4 | Available | RAM, board revision, OS, and quantity to verify |
| Requested computer | Raspberry Pi 5, 8 GB RAM | Requested | Delivery and compatible power supply to verify |
| Camera | Raspberry Pi AI Camera | Available | Ownership confirmed by the user; ribbon cable, board compatibility, mounting, camera capture, and AI operation still to verify |
| Storage | MicroSD card | To verify | Capacity, performance, and installed OS to record |
| Portable power | Power bank | To verify | Output capability, runtime, cable, and board compatibility to verify |
| Voice input | Microphone | To source or verify | Interface, capture quality, and placement to verify |
| Speech output | Headphones or speaker | To source or verify | Interface, volume control, and environmental audibility to verify |
| Wearable mounting | Glasses frame and camera accessories | To verify | Fit, attachment, field of view, and cable strain relief to check |
| Activation input | Optional pushbutton | Optional; to verify | Wiring and software integration deferred |

The camera model above follows the user's latest confirmation and supersedes
earlier generic Camera Module labels. Available equipment has not necessarily
passed an integration test.

Phase 1 runs on a development computer using a synthetic sample, an image file,
or an optional webcam. Raspberry Pi camera integration, microphone commands,
speech output, and board-specific performance work are later phases.

Run the read-only local prerequisite report from the repository root:

```sh
python scripts/check_hardware.py
python scripts/check_hardware.py --json
```

This script uses only Python's standard library. It reports software discovery
and possible camera device paths without importing optional hardware packages,
opening devices, recording audio, or making network requests. A discovered
package, command, or device path does not establish that a camera, microphone,
or speaker works; each physical device remains **unverified** in the report.

During later integration, record the board, OS, peripheral, and observed result
for each test. Evaluate the prototype in controlled conditions as a supplement
to established mobility aids. Camera imagery alone does not provide precise
distance measurements or guaranteed navigation safety.
