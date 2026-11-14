# ESP32 Ultrasonic Firmware (Glasses)

Firmware for an ESP32 Dev Module that reads two HC-SR04 ultrasonic sensors mounted on the front-left and front-right corners of a pair of glasses, fuses them into one obstacle estimate, and streams JSON over USB serial to a Raspberry Pi.

## Files
- `glasses_ultrasonic/glasses_ultrasonic.ino` - the firmware (open this folder in the Arduino IDE)

## Wiring

| Sensor | HC-SR04 pin | ESP32 pin |
|---|---|---|
| Left | VCC | VIN (5 V) |
| Left | GND | GND |
| Left | TRIG | GPIO 25 |
| Left | ECHO | voltage divider -> GPIO 34 |
| Right | VCC | VIN (5 V) |
| Right | GND | GND |
| Right | TRIG | GPIO 26 |
| Right | ECHO | voltage divider -> GPIO 35 |

ECHO outputs 5 V and ESP32 pins are 3.3 V only. Build one divider per sensor:

```
HC-SR04 ECHO --[ 1k ]--+-- ESP32 GPIO (34 or 35)
                       |
                     [ 2k ]
                       |
                      GND
```

All grounds must be connected together.

## Flashing
1. Install the ESP32 board package in the Arduino IDE.
2. Board: **ESP32 Dev Module**, upload speed 921600 (use 115200 if it fails).
3. Set `BASELINE_CM` in the code to the measured distance between the two sensor centers.
4. Upload, then open Serial Monitor at **115200 baud**.

Close the Serial Monitor before running any other program that uses the port.

## Output
One JSON object per line, about 12 per second:

```
{"seq":42,"t":51234,"l":142.3,"r":150.1,"rng":145.0,"brg":-3.2,"x":-6.1,"y":144.8,"vel":-35,"ttc":4.1,"mode":"tri","zone":"center","haz":"caution","conf":1.0}
```

| Field | Meaning |
|---|---|
| seq | line counter (detect dropped lines) |
| t | ms since boot |
| l, r | filtered left / right distance (cm) |
| rng | fused distance from the midpoint to the object (cm) |
| brg | bearing in degrees (negative = left, positive = right) |
| x, y | sideways / forward offset (cm) |
| vel | closing speed in cm/s (negative = closing) |
| ttc | time to contact in seconds, null if not closing |
| mode | `tri`, `split`, `left_only`, `right_only`, `none` |
| zone | `left`, `center`, `right` |
| haz | `clear`, `aware`, `caution`, `danger` |
| conf | confidence, 0 to 1 |

`null` means the value is unavailable. A one-time `{"event":"boot",...}` line is sent at startup; skip lines containing `event` when parsing.

Optional input: send `T22.5` (plus newline) to set the air temperature in C for speed-of-sound correction.

## Reading it on the Pi
Read `/dev/ttyUSB0` (or `/dev/ttyACM0`) at 115200 baud with pyserial and call `json.loads` on each line. Your user must be in the `dialout` group.
