"""Compile the actual sketch against local SDK stubs; never open or flash USB."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


SKETCH = Path(__file__).resolve().parents[1] / "esp32code" / "esp32code.ino"

ARDUINO = r'''
#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#define IRAM_ATTR
constexpr int LOW=0, HIGH=1, INPUT=1, OUTPUT=2, CHANGE=3;
struct portMUX_TYPE {};
#define portMUX_INITIALIZER_UNLOCKED {}
extern uint64_t fakeUs;
extern int criticalDepth, lockEntries, baud, levels[64], modes[64];
struct Trigger { int pin; uint64_t at; };
extern std::vector<Trigger> triggers;
extern std::vector<int> interruptPins;
inline void portENTER_CRITICAL(portMUX_TYPE*) { ++criticalDepth; ++lockEntries; }
inline void portEXIT_CRITICAL(portMUX_TYPE*) { assert(criticalDepth>0); --criticalDepth; }
inline void portENTER_CRITICAL_ISR(portMUX_TYPE* p) { portENTER_CRITICAL(p); }
inline void portEXIT_CRITICAL_ISR(portMUX_TYPE* p) { portEXIT_CRITICAL(p); }
inline void pinMode(int pin, int mode) { modes[pin]=mode; }
inline void digitalWrite(int pin, int level) {
  levels[pin]=level;
  if (level==HIGH) triggers.push_back({pin,fakeUs});
}
inline void delayMicroseconds(unsigned n) { fakeUs+=n; }
inline int digitalPinToInterrupt(int pin) { return pin; }
inline void attachInterruptArg(int pin, void(*)(void*), void*, int mode) {
  assert(mode==CHANGE); interruptPins.push_back(pin);
}
struct FakeSerial {
  std::string input, output;
  size_t readAt=0;
  void begin(int value) { baud=value; }
  int available() { return static_cast<int>(input.size()-readAt); }
  int read() { return available() ? static_cast<unsigned char>(input[readAt++]) : -1; }
  size_t write(const uint8_t* bytes, size_t n) {
    output.append(reinterpret_cast<const char*>(bytes), n); return n;
  }
  size_t write(uint8_t byte) { output.push_back(static_cast<char>(byte)); return 1; }
};
extern FakeSerial Serial;
'''

GPIO = r'''
#pragma once
#include "Arduino.h"
using gpio_num_t = int;
inline int gpio_get_level(gpio_num_t pin) {
  assert(criticalDepth>0); return levels[pin];
}
'''

HARNESS = r'''
#include <cstdlib>
#include <iostream>
#include "@SKETCH@"
uint64_t fakeUs=0;
int criticalDepth=0, lockEntries=0, baud=0, levels[64]={}, modes[64]={};
std::vector<Trigger> triggers;
std::vector<int> interruptPins;
FakeSerial Serial;

void check(bool condition, const char* message) {
  if (!condition) { std::cerr << message << '\n'; std::exit(2); }
}
void tick(uint64_t now) {
  check(now>=fakeUs, "test clock went backwards"); fakeUs=now; loop();
}
void feed(const std::string& text) {
  Serial.input+=text;
  while (Serial.available()) loop();
}
void edge(Sensor& sensor, int level, uint64_t now) {
  check(now>=fakeUs, "edge clock went backwards");
  fakeUs=now; levels[sensor.echo]=level; echoISR(&sensor);
}
void finish(Sensor& sensor, int width) {
  if (sensor.error) { loop(); return; }
  const uint64_t opened=fakeUs;
  if (width<0) tick(opened+ECHO_TIMEOUT_US);
  else {
    edge(sensor, HIGH, opened+100);
    edge(sensor, LOW, opened+100+static_cast<unsigned>(width));
    loop();
  }
}
void finishPair(int leftWidth, int rightWidth) {
  if (pairPhase==PairPhase::WaitLeft) tick(fakeUs+TRIGGER_GAP_US);
  check(pairPhase==PairPhase::ReadLeft, "left attempt did not start");
  finish(SL,leftWidth);
  check(pairPhase==PairPhase::WaitRight, "left attempt did not finish");
  tick(fakeUs+TRIGGER_GAP_US);
  check(pairPhase==PairPhase::ReadRight, "right attempt did not start");
  finish(SR,rightWidth);
  check(pairPhase==PairPhase::Idle, "pair did not finish");
}
void pair(const char* token, int leftWidth, int rightWidth) {
  feed(std::string("R1 ")+token+"\n"); finishPair(leftWidth,rightWidth);
}

void protocol() {
  feed("garbage\nH1 0123456789abcdeF\nR1 0123456789abcde\nR1 0123456789abcdefx\n");
  feed(std::string(80,'x')+"H1 0123456789abcdef\n");
  check(Serial.output.empty() && triggers.empty(), "malformed input triggered a reply or sensor");
  feed("H1 0123456789abcdef\nH1 fedcba9876543210\r\n");
  check(triggers.empty(), "handshake fired a sensor");
  feed("R1 1111111111111111");
  check(triggers.empty(), "partial request fired a sensor");
  feed("\n");
  check(triggers.size()==1, "new request did not start left");
  feed("H1 2222222222222222\nR1 3333333333333333\n");
  finishPair(-1,-1);
  const size_t bytes=Serial.output.size();
  tick(fakeUs+1000000);
  check(Serial.output.size()==bytes && pairPhase==PairPhase::Idle, "busy request queued another pair");
}

void timing() {
  feed("R1 0000000000000001\n");
  const uint64_t leftOpened=fakeUs;
  finish(SL,2912);
  tick(leftOpened+TRIGGER_GAP_US-1);
  check(triggers.size()==1, "right fired before global gap");
  tick(leftOpened+TRIGGER_GAP_US);
  check(triggers.size()==2 && triggers[1].pin==TRIG_R, "right did not fire after gap");
  const uint64_t rightOpened=fakeUs;
  finish(SR,5824);
  feed("R1 0000000000000002\n");
  check(triggers.size()==2 && Serial.output.find("0000000000000002")==std::string::npos,
        "second request reused a cached sample");
  tick(rightOpened+TRIGGER_GAP_US-1);
  check(triggers.size()==2, "new left fired before global gap");
  tick(rightOpened+TRIGGER_GAP_US);
  finishPair(2912,2912);
  for (size_t i=1;i<triggers.size();++i) {
    check(triggers[i].at-triggers[i-1].at>=TRIGGER_GAP_US, "trigger spacing violated");
    check(triggers[i].pin!=triggers[i-1].pin, "sensors did not alternate");
  }
}

void rawReadings() {
  pair("0000000000000001",11650,11650);
  pair("0000000000000002",2912,2912);  // A sudden closer object is not gated away.
  pair("0000000000000003",-1,-1);    // No history/coasting survives no echo.
  pair("0000000000000004",50,20000);
  pair("0000000000000005",0,117);
  levels[ECHO_L]=HIGH;
  const size_t before=triggers.size();
  pair("0000000000000006",2912,2912);
  check(triggers.size()==before+1 && triggers.back().pin==TRIG_R,
        "stuck-high echo was triggered or blocked the other channel");
}

void windows() {
  edge(SL,HIGH,10); edge(SL,LOW,1000);
  edge(SR,HIGH,1100); edge(SR,LOW,1200);
  check(!SL.complete && !SR.complete, "unarmed noise became a measurement");
  feed("R1 0000000000000001\n");
  const uint64_t opened=fakeUs;
  edge(SL,LOW,opened+20);
  check(!SL.complete && SL.armed, "fall without rise completed a pulse");
  edge(SL,HIGH,opened+100); edge(SL,LOW,opened+1100);
  check(!SL.armed && SL.complete, "first complete pulse did not disarm");
  edge(SL,HIGH,opened+1200); edge(SL,LOW,opened+1220);
  loop();
  check(leftReading.distanceMm==172, "later edge replaced first pulse");
  tick(fakeUs+TRIGGER_GAP_US);
  const uint64_t rightOpened=fakeUs;
  edge(SR,HIGH,rightOpened+100);
  edge(SR,LOW,rightOpened+ECHO_TIMEOUT_US);
  check(!SR.armed && !SR.complete && SR.timedOut, "deadline edge was accepted");
  edge(SR,HIGH,rightOpened+ECHO_TIMEOUT_US+10);
  edge(SR,LOW,rightOpened+ECHO_TIMEOUT_US+1000);
  loop();
  check(pairPhase==PairPhase::Idle, "expired channel did not complete unknown");
  pair("0000000000000002",-1,-1);
}

void timerWrap() {
  fakeUs=(uint64_t{1}<<32)-150;
  sequenceNo=UINT32_MAX;
  pair("0000000000000001",200,1000);
  pair("0000000000000002",1000,200);
}

void delayedDelivery() {
  feed("R1 0000000000000001\n");
  finish(SL,2912);
  tick(fakeUs+TRIGGER_GAP_US);
  const uint64_t opened=fakeUs;
  edge(SR,HIGH,opened+100); edge(SR,LOW,opened+1100);
  tick(fakeUs+600000);  // Main-loop delay cannot refresh the physical observation.
}

int main(int argc, char** argv) {
  check(argc==2, "scenario required");
  setup();
  check(baud==115200, "wrong serial baud");
  check(modes[25]==OUTPUT && modes[26]==OUTPUT && modes[34]==INPUT && modes[35]==INPUT,
        "pin assignment or input mode changed");
  check(interruptPins==std::vector<int>({34,35}), "echo interrupts missing");
  const std::string scenario=argv[1];
  if (scenario=="protocol") protocol();
  else if (scenario=="timing") timing();
  else if (scenario=="raw") rawReadings();
  else if (scenario=="windows") windows();
  else if (scenario=="wrap") timerWrap();
  else if (scenario=="delayed") delayedDelivery();
  else check(false,"unknown scenario");
  check(criticalDepth==0 && lockEntries>0, "critical section not balanced/exercised");
  std::cout << Serial.output;
}
'''


@pytest.fixture(scope="module")
def firmware_binary(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A local C++ compiler is needed; this test never installs one.")
    directory = tmp_path_factory.mktemp("ultrasonic-native")
    (directory / "driver").mkdir()
    (directory / "Arduino.h").write_text(ARDUINO, encoding="utf-8")
    (directory / "driver" / "gpio.h").write_text(GPIO, encoding="utf-8")
    (directory / "esp_timer.h").write_text(
        '#pragma once\n#include "Arduino.h"\ninline int64_t esp_timer_get_time() { return static_cast<int64_t>(fakeUs); }\n',
        encoding="utf-8",
    )
    (directory / "esp_random.h").write_text(
        '#pragma once\n#include <cstdint>\ninline uint32_t esp_random() { return 0xabc; }\n', encoding="utf-8",
    )
    source = directory / "harness.cpp"
    source.write_text(HARNESS.replace("@SKETCH@", SKETCH.as_posix()), encoding="utf-8")
    binary = directory / "firmware-tests"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(directory), str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=30,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return binary


def run_scenario(binary, name):
    completed = subprocess.run([str(binary), name], capture_output=True, text=True, timeout=5)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = completed.stdout.splitlines(keepends=True)
    assert lines
    assert all(line.endswith("\n") and len(line.encode("ascii")) <= 512 for line in lines)
    messages = [json.loads(line) for line in lines]
    for message in messages:
        assert message["v"] == 1
        assert re.fullmatch(r"[0-9a-f]{8}", message["boot"])
        assert re.fullmatch(r"[0-9a-f]{16}", message["id"])
        if message["type"] == "sample":
            assert set(message) == {"v", "type", "id", "boot", "seq", "left", "right"}
            assert type(message["seq"]) is int and 0 <= message["seq"] <= 2**32 - 1
            for side in ("left", "right"):
                reading = message[side]
                assert set(reading) == {"status", "distance_mm", "age_ms"}
                assert type(reading["age_ms"]) is int and reading["age_ms"] >= 0
                assert reading["status"] in {"ok", "no_echo", "out_of_range", "error"}
                if reading["status"] == "ok":
                    assert type(reading["distance_mm"]) is int and 20 <= reading["distance_mm"] <= 3000
                else:
                    assert reading["distance_mm"] is None
    return messages


def test_handshake_commands_reject_busy_and_do_not_queue(firmware_binary):
    messages = run_scenario(firmware_binary, "protocol")
    assert [message["type"] for message in messages] == ["hello", "hello", "error", "error", "sample"]
    assert messages[0] == {
        "v": 1, "type": "hello", "id": "0123456789abcdef", "fw": "vidi_ultrasonic",
        "boot": "00000abc", "channels": ["left", "right"],
    }
    assert [message["id"] for message in messages[2:]] == ["2" * 16, "3" * 16, "1" * 16]
    assert all(message["code"] == "busy" for message in messages[2:4])
    assert messages[-1]["seq"] == 0
    assert messages[-1]["left"]["status"] == messages[-1]["right"]["status"] == "no_echo"


def test_new_pair_timing_is_global_and_each_channel_keeps_its_age(firmware_binary):
    first, second = run_scenario(firmware_binary, "timing")
    assert [first["seq"], second["seq"]] == [0, 1]
    assert [first["id"], second["id"]] == ["0" * 15 + "1", "0" * 15 + "2"]
    assert first["left"]["distance_mm"] == second["left"]["distance_mm"] == 500
    assert first["right"]["distance_mm"] == 1000
    assert second["right"]["distance_mm"] == 500
    assert first["left"]["age_ms"] >= 100 and first["right"]["age_ms"] == 0


def test_raw_closer_step_and_missing_echo_never_reuse_history(firmware_binary):
    far, near, missing, outside, zero, stuck = run_scenario(firmware_binary, "raw")
    assert far["left"]["distance_mm"] == 2000
    assert near["left"]["distance_mm"] == 500
    assert missing["left"]["distance_mm"] is missing["right"]["distance_mm"] is None
    assert missing["left"]["status"] == missing["right"]["status"] == "no_echo"
    assert outside["left"]["status"] == outside["right"]["status"] == "out_of_range"
    assert zero["left"]["status"] == "error" and zero["right"]["distance_mm"] == 20
    assert stuck["left"]["status"] == "error" and stuck["right"]["distance_mm"] == 500


def test_armed_first_pulse_and_deadline_windows(firmware_binary):
    first, second = run_scenario(firmware_binary, "windows")
    assert first["left"]["distance_mm"] == 172
    assert first["right"]["status"] == "no_echo"
    assert second["left"]["status"] == second["right"]["status"] == "no_echo"


def test_microsecond_and_sequence_wrap_are_well_defined(firmware_binary):
    first, second = run_scenario(firmware_binary, "wrap")
    assert [first["seq"], second["seq"]] == [2**32 - 1, 0]
    assert [first["left"]["distance_mm"], first["right"]["distance_mm"]] == [34, 172]
    assert [second["left"]["distance_mm"], second["right"]["distance_mm"]] == [172, 34]


def test_delayed_serialization_does_not_refresh_measurement_age(firmware_binary):
    message, = run_scenario(firmware_binary, "delayed")
    assert message["left"]["age_ms"] >= 700
    assert message["right"]["age_ms"] == 600
