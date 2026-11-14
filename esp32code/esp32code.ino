/*
  vidi ultrasonic firmware: classic ESP32 Dev Module, Arduino-ESP32 3.3.12.
  Left TRIG/ECHO: GPIO25/34. Right: GPIO26/35. Each HC-SR04 needs a separate
  5 V -> 3.3 V echo divider and a common ground; see README.md.

  USB 115200: H1 <16 lowercase hex> identifies this firmware.
  R1 <16 lowercase hex> requests a NEW left/right pair. No free-running
  stream, prediction, triangulation, TTC, confidence, or clearance claims.
  Interrupt acquisition, >=100 ms global trigger gap, 30 ms echo deadline.
  Sequential triggers reduce but cannot guarantee absence of crosstalk.
*/

#include <Arduino.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "driver/gpio.h"
#include "esp_random.h"
#include "esp_timer.h"

constexpr uint8_t TRIG_L = 25, ECHO_L = 34;
constexpr uint8_t TRIG_R = 26, ECHO_R = 35;
constexpr uint32_t ECHO_TIMEOUT_US = 30000;
constexpr uint32_t TRIGGER_GAP_US = 100000;
constexpr int MIN_MM = 20, MAX_MM = 3000;
constexpr size_t COMMAND_CAPACITY = 64;
constexpr size_t OUTPUT_CAPACITY = 512;
constexpr float MM_PER_US = (331.3f + 0.606f * 20.0f) / 2000.0f;

portMUX_TYPE echoMux = portMUX_INITIALIZER_UNLOCKED;

struct Sensor {
  uint8_t trig, echo;
  volatile bool armed = false;
  volatile bool rose = false;
  volatile bool complete = false;
  volatile bool error = false;
  volatile bool timedOut = false;
  volatile uint32_t openedUs = 0;
  volatile uint32_t riseUs = 0;
  volatile uint32_t fallUs = 0;
  Sensor(uint8_t triggerPin, uint8_t echoPin) : trig(triggerPin), echo(echoPin) {}
};

struct Reading {
  const char* status = "error";
  int distanceMm = -1;
  uint32_t observedUs = 0;
};

enum class PairPhase { Idle, WaitLeft, ReadLeft, WaitRight, ReadRight };
Sensor SL(TRIG_L, ECHO_L), SR(TRIG_R, ECHO_R);
Reading leftReading, rightReading;
PairPhase pairPhase = PairPhase::Idle;
char requestId[17] = {};
char bootId[9] = {};
uint32_t sequenceNo = 0;
uint32_t lastSlotUs = 0;
bool haveSlot = false;

// Every capture field is read/written under this same mux in main and ISR.
void IRAM_ATTR echoISR(void* argument) {
  Sensor* sensor = static_cast<Sensor*>(argument);
  portENTER_CRITICAL_ISR(&echoMux);
  const uint32_t now = static_cast<uint32_t>(esp_timer_get_time());
  if (sensor->armed) {
    if (static_cast<uint32_t>(now - sensor->openedUs) >= ECHO_TIMEOUT_US) {
      sensor->armed = false;
      sensor->timedOut = true;
    } else if (gpio_get_level(static_cast<gpio_num_t>(sensor->echo))) {
      if (!sensor->rose) {
        sensor->riseUs = now;
        sensor->rose = true;
      }
    } else if (sensor->rose) {
      sensor->fallUs = now;
      sensor->complete = true;
      sensor->armed = false;  // Freeze FIRST complete pulse, including zero width.
    }
  }
  portEXIT_CRITICAL_ISR(&echoMux);
}

void startChannel(Sensor& sensor) {
  portENTER_CRITICAL(&echoMux);
  sensor.armed = false;
  sensor.rose = false;
  sensor.complete = false;
  sensor.timedOut = false;
  sensor.riseUs = 0;
  sensor.fallUs = 0;
  sensor.openedUs = static_cast<uint32_t>(esp_timer_get_time());
  sensor.error = gpio_get_level(static_cast<gpio_num_t>(sensor.echo)) != 0;
  const bool blocked = sensor.error;
  portEXIT_CRITICAL(&echoMux);

  if (!blocked) {
    digitalWrite(sensor.trig, LOW);
    delayMicroseconds(2);
    digitalWrite(sensor.trig, HIGH);
    delayMicroseconds(10);
    portENTER_CRITICAL(&echoMux);
    digitalWrite(sensor.trig, LOW);
    sensor.openedUs = static_cast<uint32_t>(esp_timer_get_time());
    // A line already high cannot be associated with this trigger safely.
    sensor.error = gpio_get_level(static_cast<gpio_num_t>(sensor.echo)) != 0;
    sensor.armed = !sensor.error;
    portEXIT_CRITICAL(&echoMux);
  }
  // Pace failed attempts too; this also guarantees >=100 ms between triggers.
  lastSlotUs = static_cast<uint32_t>(esp_timer_get_time());
  haveSlot = true;
}

bool finishChannel(Sensor& sensor, Reading& reading) {
  bool complete, error, timedOut;
  uint32_t opened, rise, fall;
  const uint32_t now = static_cast<uint32_t>(esp_timer_get_time());
  portENTER_CRITICAL(&echoMux);
  if (sensor.armed && static_cast<uint32_t>(now - sensor.openedUs) >= ECHO_TIMEOUT_US) {
    sensor.armed = false;
    sensor.timedOut = true;
  }
  complete = sensor.complete;
  error = sensor.error;
  timedOut = sensor.timedOut;
  opened = sensor.openedUs;
  rise = sensor.riseUs;
  fall = sensor.fallUs;
  portEXIT_CRITICAL(&echoMux);
  if (!complete && !error && !timedOut) return false;

  reading.distanceMm = -1;
  if (error) {
    reading.status = "error";
    reading.observedUs = opened;
  } else if (!complete) {
    reading.status = "no_echo";
    reading.observedUs = opened + ECHO_TIMEOUT_US;
  } else {
    reading.observedUs = fall;
    const uint32_t width = fall - rise;  // Unsigned subtraction handles timer wrap.
    if (!width) {
      reading.status = "error";
    } else {
      const int mm = static_cast<int>(width * MM_PER_US + 0.5f);
      if (mm < MIN_MM || mm > MAX_MM) {
        reading.status = "out_of_range";
      } else {
        reading.status = "ok";
        reading.distanceMm = mm;
      }
    }
  }
  return true;
}

void emitLine(const char* line, int length) {
  // Include the sole newline in the protocol's 512-byte limit.
  if (length > 0 && static_cast<size_t>(length) < OUTPUT_CAPACITY) {
    Serial.write(reinterpret_cast<const uint8_t*>(line), static_cast<size_t>(length));
    Serial.write(static_cast<uint8_t>('\n'));
  }
}

void emitHello(const char* token) {
  char line[OUTPUT_CAPACITY];
  const int n = snprintf(line, sizeof(line),
      "{\"v\":1,\"type\":\"hello\",\"id\":\"%s\",\"fw\":\"vidi_ultrasonic\","
      "\"boot\":\"%s\",\"channels\":[\"left\",\"right\"]}", token, bootId);
  emitLine(line, n);
}

void emitBusy(const char* token) {
  char line[OUTPUT_CAPACITY];
  const int n = snprintf(line, sizeof(line),
      "{\"v\":1,\"type\":\"error\",\"id\":\"%s\",\"boot\":\"%s\",\"code\":\"busy\"}", token, bootId);
  emitLine(line, n);
}

void emitSample() {
  const uint32_t now = static_cast<uint32_t>(esp_timer_get_time());
  char leftMm[12], rightMm[12], line[OUTPUT_CAPACITY];
  if (leftReading.distanceMm < 0) strcpy(leftMm, "null");
  else snprintf(leftMm, sizeof(leftMm), "%d", leftReading.distanceMm);
  if (rightReading.distanceMm < 0) strcpy(rightMm, "null");
  else snprintf(rightMm, sizeof(rightMm), "%d", rightReading.distanceMm);
  const int n = snprintf(line, sizeof(line),
      "{\"v\":1,\"type\":\"sample\",\"id\":\"%s\",\"boot\":\"%s\",\"seq\":%lu,"
      "\"left\":{\"status\":\"%s\",\"distance_mm\":%s,\"age_ms\":%lu},"
      "\"right\":{\"status\":\"%s\",\"distance_mm\":%s,\"age_ms\":%lu}}",
      requestId, bootId, static_cast<unsigned long>(sequenceNo++),
      leftReading.status, leftMm, static_cast<unsigned long>(static_cast<uint32_t>(now - leftReading.observedUs) / 1000),
      rightReading.status, rightMm, static_cast<unsigned long>(static_cast<uint32_t>(now - rightReading.observedUs) / 1000));
  emitLine(line, n);
}

void acceptCommand(const char* line, size_t length) {
  if (length && line[length - 1] == '\r') --length;  // Accept LF or CRLF.
  if (length != 19 || (line[0] != 'H' && line[0] != 'R') || line[1] != '1' || line[2] != ' ') return;
  for (size_t i = 3; i < 19; ++i) {
    if (!((line[i] >= '0' && line[i] <= '9') || (line[i] >= 'a' && line[i] <= 'f'))) return;
  }
  char token[17];
  memcpy(token, line + 3, 16);
  token[16] = 0;
  if (pairPhase != PairPhase::Idle) {
    emitBusy(token);
  } else if (line[0] == 'H') {
    emitHello(token);
  } else {
    memcpy(requestId, token, sizeof(requestId));
    leftReading = Reading{};
    rightReading = Reading{};
    pairPhase = PairPhase::WaitLeft;
  }
}

void handleSerial() {
  static char command[COMMAND_CAPACITY];
  static size_t used = 0;
  static bool discard = false;
  // Bound serial work per loop so partial/flooded input cannot starve acquisition.
  for (size_t count = 0; count < COMMAND_CAPACITY && Serial.available(); ++count) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      if (!discard) acceptCommand(command, used);
      used = 0;
      discard = false;
    } else if (!discard) {
      if (used < sizeof(command) - 1) command[used++] = c;
      else { used = 0; discard = true; }
    }
  }
}

void servicePair() {
  const uint32_t now = static_cast<uint32_t>(esp_timer_get_time());
  const bool mayTrigger = !haveSlot || static_cast<uint32_t>(now - lastSlotUs) >= TRIGGER_GAP_US;
  switch (pairPhase) {
    case PairPhase::Idle: break;
    case PairPhase::WaitLeft:
      if (mayTrigger) { startChannel(SL); pairPhase = PairPhase::ReadLeft; }
      break;
    case PairPhase::ReadLeft:
      if (finishChannel(SL, leftReading)) pairPhase = PairPhase::WaitRight;
      break;
    case PairPhase::WaitRight:
      if (mayTrigger) { startChannel(SR); pairPhase = PairPhase::ReadRight; }
      break;
    case PairPhase::ReadRight:
      if (finishChannel(SR, rightReading)) { emitSample(); pairPhase = PairPhase::Idle; }
      break;
  }
}

void setup() {
  Serial.begin(115200);
  snprintf(bootId, sizeof(bootId), "%08lx", static_cast<unsigned long>(esp_random()));
  Sensor* sensors[2] = { &SL, &SR };
  for (Sensor* sensor : sensors) {
    pinMode(sensor->trig, OUTPUT);
    digitalWrite(sensor->trig, LOW);
    pinMode(sensor->echo, INPUT);  // GPIO34/35 have no internal pulls.
    attachInterruptArg(digitalPinToInterrupt(sensor->echo), echoISR, sensor, CHANGE);
  }
}

void loop() {
  handleSerial();
  servicePair();
}
