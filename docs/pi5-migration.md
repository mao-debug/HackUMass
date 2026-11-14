# Move the existing vidi SD card to Pi 5

The existing Pi 4 microSD now boots a **Raspberry Pi 5 Model B Rev 1.1**. The
supplied report confirms Debian 12/Bookworm, kernel
`6.12.115+rpt-rpi-2712`, aarch64 with a 64-bit Python process, and Python 3.11.2
in `/home/simaofea/vidi/.venv-pi/bin/python`. Kernel-visible memory is 8061 MiB,
consistent with an 8 GB board; exact physical capacity is not independently
verified. IMX708 enumerates at camera index 0. The camera-only preview captured
282 frames at a measured **21.79 FPS** with detector `none` and returned normally
after q. A subsequent Gemini live run captured 684 frames at **23.43 FPS**,
with four HTTP 200 replies, four successful analyses, and zero unavailable
results. The user separately confirmed hearing the local eSpeak test word on
Pi 5. **Image colors/clarity, recognition accuracy, power/thermal behavior,
local-model inference, and integrated OCR/audio remain pending.**

Discovery reports NumPy 1.24.2, PyYAML 6, setuptools 66.1.1, Google SDK
2.34.0, and Picamera2 0.3.31. The earlier Pi 4 record had Picamera2 0.3.27 and
OpenCV 4.6.0. The completed Pi 5 application run now demonstrates usable native
capture, OpenCV/NumPy processing, and preview. Its exact imported OpenCV version
is still unreported. Null `opencv-python` package metadata does not mean apt's
OpenCV module is absent. No installation or update is indicated by this evidence.

Raspberry Pi's [Pi 5 FAQ](https://www.raspberrypi.com/products/raspberry-pi-5/)
supports Bookworm, and the current
[legacy 64-bit OS listing](https://www.raspberrypi.com/software/operating-systems/)
lists both Pi 4 and Pi 5 with Bookworm/kernel 6.12. This supports trying the
existing installation; it does not establish compatibility of every installed
package or custom boot setting. Inspect first. Do not reimage, upgrade the OS,
replace Python packages, or edit firmware settings as a migration prerequisite.

Keep the repository, `.venv-pi`, local models, artifacts, and
`~/.config/vidi/gemini.key` on the existing card. Do not print, paste, or commit
the saved key. An optional `.venv-tts` and Piper voice download remain subject
to separate approval; they are not needed for migration.

## Check the physical connections

- Shut down and disconnect power before moving the SD card or camera ribbon.
- Camera Module 3 has a 15-pin connector; Pi 5 has mini 22-pin camera/display
  connectors. Verify a **Standard–Mini 15-to-22-pin camera cable**, seated and
  latched at both ends. The old Pi 4 15-to-15-pin ribbon does not fit directly.
  Follow the official [camera connection guide](https://www.raspberrypi.com/documentation/accessories/camera-software.html#connect-the-camera)
  for contact orientation. Do not force the connector.
- Record the actual USB-C supply and cable. Raspberry Pi recommends its 27 W
  supply for Pi 5. A compatible 3 A supply can boot it but limits USB peripheral
  current to 600 mA; a detected 5 V/5 A supply allows 1.6 A. A power bank's total
  wattage alone does not verify that profile. See the official
  [power requirements](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#power-supply).
- Record the case/heatsink/fan and leave airflow unobstructed. Sustained work
  can throttle an uncooled Pi 5; verify cooling before longer trials. The
  official [cooling explanation](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/)
  describes the load-dependent behavior. No new purchase is assumed.

## Inspect this boot without changing the installation

From the Pi desktop terminal, enter the repository before using relative
paths. The earlier `.venv-pi/bin/python: No such file` occurred in `~` and was
resolved with `cd ~/vidi`; it did not require rebuilding the environment.
These commands can record or repeat the identity checks:

```sh
cd ~/vidi
tr -d '\000' < /sys/firmware/devicetree/base/model
cat /etc/os-release
uname -r
uname -m
getconf LONG_BIT
/usr/bin/python3 --version
free -h
df -h /
PYTHONNOUSERSITE=1 .venv-pi/bin/python scripts/check_hardware.py --json
```

The model now identifies Pi 5; preserve the supplied report and record remaining
free-space observations. OS-visible memory is not the same as the board's
advertised RAM capacity. The diagnostic discovers package origins,
versions where available, and commands without importing optional hardware
packages or opening devices. Discovery does not certify that those packages
work on the new board.

Check the retained application environment separately:

```sh
.venv-pi/bin/python --version
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --version
command -v rpicam-hello libcamera-hello vcgencmd bluetoothctl pactl wpctl espeak-ng paplay pw-play aplay
```

The supplied report finds rpicam, vcgencmd, Bluetooth/audio tools, and eSpeak NG.
Piper is absent from the camera environment and its downloads remain pending;
it would use its own optional `.venv-tts`. Missing local Tesseract and optional
microphone modules do not block the existing Gemini/camera workflow.

If a runtime or required package is missing, stop and record the exact error.
Reuse existing package approvals where applicable; request approval before new
packages, download sources, or upgrades outside that scope. Do not run desktop
pip requirements on the Pi, rebuild `.venv-pi`, or perform a blanket apt upgrade.

## Enumerate the camera, then preview without AI

Close other camera applications. If `rpicam-hello` is available:

```sh
rpicam-hello --list-cameras
```

If only the older command is available, use
`libcamera-hello --list-cameras` instead. Confirm the sensor and camera index;
the migrated Camera Module 3 should enumerate as IMX708. An empty list calls
for checking the powered-off ribbon connection and actual installed stack
before attempting AI. The official
[camera enumeration reference](https://www.raspberrypi.com/documentation/accessories/camera-software.html#list-cameras)
describes the output. The supplied Pi 5 enumeration already reports camera 0
as IMX708 at 4608×2592, under
`/base/axi/pcie@1000120000/rp1/i2c@80000/imx708@1a`. These are sensor-enumeration
details, not delivered capture dimensions or proof of successful capture.

Then run this local camera-only check:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-live.yaml --continuous --detector none --preview --output artifacts/pi5-camera.png --report artifacts/pi5-camera.json
```

`--detector none` overrides the profile's Gemini detector. This command makes
no uploads and reads no saved API key. It requests 640×480 at 30 FPS; actual
performance must be measured. Inspect colors, focus, orientation, and whether
the preview updates smoothly. Press q/Esc or Ctrl+C to stop and release the
camera; Ctrl+C normally returns status 130. The output is a final still image,
not a video recording.

The user has completed this exact command: 282 frames, 21.79 measured FPS,
0 successful analyses and 0 unavailable results, as expected with detector
`none`. It returned to the terminal prompt after q without a traceback or fatal
error. The log identifies libcamera `0.5.2+99-bfd68f78`, libpisp v1.2.1
(`981977ff21f3`), tuning `rpi/pisp/imx708.json`, and the `BCM2712_D0` PiSP
pipeline with CFE `/dev/media3` and ISP `/dev/media1`. Main and raw streams were
640×480 RGB888 and 1536×864 PISP_COMP1. This validates the exercised capture/
preview path, not recognition or a guaranteed 30 FPS rate.

Repeated libpng `iCCP: known incorrect sRGB profile` warnings and one
GLib-GObject `g_object_unref` critical were printed; they were nonfatal in this
completed run. Their origins are unconfirmed, so these messages alone do not
justify reinstalling camera drivers. Image colors/clarity remain unanswered.
The command requested the image/report exports above with no reported export
error; their contents have not been independently inspected.

If `vcgencmd` was found, record these before and after the preview:

```sh
vcgencmd measure_temp
vcgencmd get_throttled
```

Retain the exact warning/status output. `get_throttled` distinguishes current
and historical undervoltage/throttling flags; a nonzero value needs decoding,
not an automatic firmware override. See the official
[`vcgencmd` reference](https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd).
If the command is absent, report that fact without installing it automatically.

## Recheck Bluetooth and local audio

The Pi 5 has a different physical Bluetooth controller. Moving the SD card
does not establish a headphone connection on that controller. Recheck pairing
and connect through the desktop Bluetooth menu; then choose the headphones
from the speaker icon's output menu and check mute/comfortable volume. Record
`bluetoothctl show` if that command is available. Pair again if the old pairing
does not connect; do not delete system Bluetooth state as a first step.

Follow [Phase 3 audio checks](phase3.md#bluetooth-headphones-on-the-pi) and
Raspberry Pi's [audio guide](https://pip-assets.raspberrypi.com/categories/1259-audio-camera-and-display/documents/RP-008124-WP/Choosing-an-Audio-option.pdf).
The user reported audible speech on the previous board but described its voice
as robotic. On the Pi 5, the user subsequently confirmed hearing
`espeak-ng vidi` after the headphone/output check. This passes the basic local
audio test; integrated OCR read-aloud and voice quality are separate checks.
The current report finds eSpeak NG, paplay, pw-play, aplay, pactl, and wpctl;
their presence does not verify an active server, selected output, or audibility.
eSpeak NG installation was approved. Missing diagnostic tools do not authorize
replacing the audio server.

## Resume one capability at a time

After the camera-only check, optional local CPU detection uses the existing
approved assets without uploads:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi.yaml --max-frames 20 --preview --report artifacts/pi5-local.json
```

This temporary 20-class MobileNet profile requests 2 FPS; its rate cap is not
a maximum-speed benchmark. No model download or package installation occurs.

For an explicitly requested Gemini live run, use the existing saved key and
approved upload policy:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-live.yaml --continuous --gemini-live --preview --report artifacts/pi5-gemini-live.json
```

This authorizes repeated selected images, with one request at a time and at
least five seconds after completion before the next request. Errors pause
uploads; q/Esc or Ctrl+C stops new submissions. Pi 5 does not guarantee faster
cloud responses. See [Gemini behavior and credentials](gemini.md).

Stop live capture before a separate one-image reading:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --config config/pi-gemini-ocr.yaml --gemini-read --report artifacts/pi5-gemini-ocr.json
```

Add `--speak` only for an explicit local speech request after headphone output
works. No speech-generation API is used. Keep OCR recognition, speech-engine
completion, and actual audibility as separate acceptance results.

## Pi 5 acceptance record

Identity/discovery, enumeration, camera-only preview, Gemini live identification,
and basic local speech now have supplied evidence. Other checks remain pending.
The full record is in
[hardware.md](hardware.md#pi-5-migration-record). Earlier Pi 4 measurements do
not transfer to Pi 5.

| Check | Status |
| --- | --- |
| Board identity and runtime | Confirmed Pi 5 Model B Rev 1.1, Debian 12/Bookworm, kernel `6.12.115+rpt-rpi-2712`, aarch64/64-bit, Python 3.11.2 in retained venv |
| Memory | 8061 MiB kernel-visible, consistent with 8 GB; physical capacity not independently verified |
| Package discovery/native operation | Reported versions/origins; usable capture, OpenCV/NumPy processing, and preview demonstrated; exact OpenCV version unreported |
| Local model assets | Retained from Pi 4; Pi 5 load/inference pending |
| IMX708 enumeration | Confirmed index 0 at 4608×2592; physical cable inspection not recorded |
| Camera-only preview/rate/completion | Passed: 282 frames, measured 21.79 FPS, normal return after q; detector `none` |
| Image quality and exports | Colors/clarity pending; image/report exports requested without reported error, contents not independently inspected |
| Supply/cable, cooling, temperatures and throttle flags | Pending |
| Audible local speech | Passed by user confirmation: heard `espeak-ng vidi` on Pi 5 after checking headphones/output |
| Integrated OCR read-aloud and neural voice | Pending |
| Gemini live identification | Passed: 684 frames, measured 23.43 FPS, 4 successful responses/HTTP 200, 0 unavailable; prediction accuracy not independently verified |
| Optional local detection and Gemini OCR read-aloud | Pending |
