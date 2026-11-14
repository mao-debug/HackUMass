# Move the existing vidi SD card to Pi 5

The user reports moving the working Pi 4 microSD card to a new Pi 5. The
card's known baseline is 64-bit Bookworm/Debian 12, aarch64, kernel 6.12,
Python 3.11.2, Picamera2 0.3.27, and OS OpenCV 4.6.0. **Pi 5 board/RAM,
camera, power, cooling, audio, and application acceptance checks are pending.**
The originally requested 8 GB capacity is not a verified property of this board.

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

From the Pi desktop terminal and repository root, run:

```sh
tr -d '\000' < /sys/firmware/devicetree/base/model
cat /etc/os-release
uname -r
uname -m
getconf LONG_BIT
/usr/bin/python3 --version
free -h
df -h /
/usr/bin/python3 scripts/check_hardware.py --json
```

The model should identify Pi 5; record the actual text, OS, architecture,
Python bitness, memory, and free space. OS-visible memory is not the same as
the board's advertised RAM capacity. The diagnostic discovers package origins,
versions where available, and commands without importing optional hardware
packages or opening devices. Discovery does not certify that those packages
work on the new board.

Check the retained application environment separately:

```sh
.venv-pi/bin/python --version
PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-pi/bin/python -m vidi --version
command -v rpicam-hello libcamera-hello vcgencmd bluetoothctl pactl wpctl espeak-ng paplay pw-play
```

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
describes the output.

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
as robotic. Pi 5 headphone routing and playback must be checked again.
eSpeak NG installation is approved. Missing diagnostic tools do not authorize
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

All entries remain pending until new evidence is supplied. Earlier Pi 4
measurements remain in [hardware.md](hardware.md) and do not transfer to Pi 5.

| Check | Status |
| --- | --- |
| Board identity, revision/RAM, OS/kernel, architecture/Python | Pending |
| Retained environment, package origins, local model assets | Pending |
| Standard–Mini ribbon and IMX708 enumeration | Pending |
| Camera-only preview, saved image, actual rate and cleanup | Pending |
| Supply/cable, cooling, temperatures and throttle flags | Pending |
| Bluetooth pairing, selected output and audible local speech | Pending |
| Optional local detection and explicit Gemini runs | Pending |
