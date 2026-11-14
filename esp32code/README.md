# vidi ESP32 ultrasonic firmware

The sketch **esp32code/esp32code.ino** adapts the partner's two-HC-SR04,
interrupt-based acquisition for request/response USB serial. It reports fresh
**left** and **right** observations. It does not triangulate a center, estimate
TTC, predict through dropouts, assign confidence, or report a clear path.
No echo is unknown.

## Board and wiring

Target: **classic ESP32 Dev Module**, Arduino-ESP32 core **3.3.12**. Verify the
actual board pinout; other ESP32 families are not interchangeable targets.

| Sensor | VCC | GND | TRIG | ECHO |
| --- | --- | --- | --- | --- |
| Left | Verified 5 V supply | Common ground | GPIO25 | Divider to GPIO34 |
| Right | Verified 5 V supply | Common ground | GPIO26 | Divider to GPIO35 |

Verify the particular DevKit's VIN/5 V pin supplies the required voltage in its
chosen power arrangement. HC-SR04 echo is a 5 V signal: do not connect it
directly to ESP32 GPIO. Keep one divider per sensor:

~~~text
HC-SR04 ECHO --[1k ohm]--+-- GPIO34 or GPIO35
                       |
                    [2k ohm]
                       |
                      GND
~~~

The nominal output is about 3.33 V for a 5 V echo. Verify resistor values,
actual echo voltage, and board limits before connection. GPIO34/35 are
input-only without internal pull-up/down resistors; the divider provides an
external ground path. Do not omit protection based solely on a replacement
module's product name; verify its electrical specifications.

## Protocol version 1

115200 baud, newline-delimited ASCII requests and JSON replies. Commands use
an exactly 16-character lowercase hexadecimal token. LF and CRLF input are
accepted. Commands are bounded to 64 bytes including newline. Oversized or
malformed lines are discarded through the next newline without reflection.
Replies are at most 512 bytes including a single LF.

Identify the firmware while idle:

~~~text
H1 0123456789abcdef
~~~

~~~json
{"v":1,"type":"hello","id":"0123456789abcdef","fw":"vidi_ultrasonic","boot":"89abcdef","channels":["left","right"]}
~~~

The boot ID is random eight-character lowercase hexadecimal, generated once
per boot, not a MAC address. The host must validate this identity rather than
accepting arbitrary ESP32 log lines. ROM startup noise may still occur.

Request a **new pair after the command**, using a new token:

~~~text
R1 fedcba9876543210
~~~

~~~json
{"v":1,"type":"sample","id":"fedcba9876543210","boot":"89abcdef","seq":0,"left":{"status":"ok","distance_mm":500,"age_ms":100},"right":{"status":"no_echo","distance_mm":null,"age_ms":0}}
~~~

There is no cached-sample response or unsolicited sampling stream. One pair
is in flight at a time. A valid H1/R1 received while busy produces an error
with that command's token; it does not replace or queue the original request:

~~~json
{"v":1,"type":"error","id":"0123456789abcdef","boot":"89abcdef","code":"busy"}
~~~

The uint32 sample sequence wraps at 2^32; hello/error replies do not advance
it. Each channel contains:

| Field | Meaning |
| --- | --- |
| status: ok | First complete pulse from this armed measurement, converted at fixed 20 C |
| status: no_echo | No complete pulse before the 30 ms deadline |
| status: out_of_range | Pulse rounds outside the supported 20–3000 mm interval |
| status: error | Echo already high at arming, or a zero-width pulse |
| distance_mm | Integer 20–3000 only for ok; otherwise JSON null |
| age_ms | Time since that channel's pulse completion or failed-attempt completion, at serialization |

The first complete pulse freezes under a critical section. Later edges and
unarmed/deadline-expired edges cannot replace it. L/R triggers are at least
100 ms apart globally, including across requests. Repeated requests can
therefore acquire about five pairs per second, with sequential observations
roughly 100 ms apart. This reduces acoustic interference but does not guarantee
its absence. Ages include actual delay rather than refreshing old values.
The Pi must additionally enforce request timeouts and freshness limits.

## Build and bench verification

The approved local target cross-compile passed using **Arduino CLI 1.6.0** and
**Espressif Arduino-ESP32 3.3.12**, with all warnings enabled and no warnings.
Target settings
were generic ESP32 Dev Module, DIO flash at 40 MHz, 4 MB flash, PSRAM disabled,
and upload speed 115200. This verifies compilation, not the connected board's
identity or wiring. No GPIO or USB serial device was opened and nothing was
flashed.

The successful clean rebuild reported **276476 program bytes (21%)** and
**22788 static RAM bytes (6%)**.

The retained local tools/config are under ignored **artifacts/tools/arduino/**.
The reproducible command from the repository root is:

~~~sh
artifacts/tools/arduino/arduino-cli --config-file artifacts/tools/arduino/arduino-cli.yaml compile --fqbn esp32:esp32:esp32:FlashMode=dio,FlashFreq=40,FlashSize=4M,PSRAM=disabled,UploadSpeed=115200 --warnings all --build-path artifacts/firmware-build --output-dir artifacts/firmware esp32code
~~~

Build intermediates are in **artifacts/firmware-build/**. Local-only outputs
include **artifacts/firmware/esp32code.ino.bin** and the corresponding merged,
bootloader, partition, and ELF files. These artifacts are ignored and not
committed; source checkout alone does not include a prebuilt image. In Arduino
IDE, open the esp32code folder/sketch and select the same core/target settings.
Close Serial Monitor before the Pi opens the same USB port. Firmware does not
change Linux permissions or install a serial reader.

The native tests in **tests/test_ultrasonic_firmware.py** compile the actual
sketch with temporary SDK stubs using an already installed host compiler.
The six tests passed and exercise protocol, capture windows, timing, and
failures without installing an ESP32 core, opening USB devices, or flashing a
board. These native SDK simulations are separate from the successful real
ESP32 target compile and the still-pending physical electrical/acoustic checks.

Before claiming a working sensor path, record target build/upload, actual board
and protection wiring, left/right identity, reference distances, valid-return
rates, no-echo/unplug behavior, closer-target steps, repeated requests, and
serial disconnect/reconnect behavior. Test each sensor separately and both
together. No physical accuracy, crosstalk, wearable runtime, obstacle coverage,
or navigation-safety result is claimed here.
