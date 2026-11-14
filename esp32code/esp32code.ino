/*
  glasses_ultrasonic.ino  -  ESP32 Dev Module
  Two HC-SR04 sensors on the front-left / front-right corners of a pair of glasses.
  Fuses both ranges into one obstacle estimate and streams newline-delimited JSON
  over USB serial (115200 baud) to a Raspberry Pi.

  WIRING (HC-SR04 runs on 5 V, ESP32 GPIO is 3.3 V and NOT 5 V tolerant):
    VCC  -> ESP32 VIN (5 V)         GND -> GND
    TRIG -> GPIO directly (3.3 V is enough to trigger)
    ECHO -> 1k ohm -> GPIO node -> 2k ohm -> GND   (divider: 5 V -> ~3.3 V)
    (or use 3.3 V-native HC-SR04P / RCWL-1601 modules and skip the divider)

    Left  sensor: TRIG=GPIO25  ECHO=GPIO34
    Right sensor: TRIG=GPIO26  ECHO=GPIO35

  PIPELINE
    1. Sensors fire alternately (never together) -> no acoustic cross-talk.
    2. Echo width measured by pin-change interrupt (non-blocking, microsecond timing).
    3. Per sensor: median-of-3 spike rejection -> alpha-beta tracker
       (smooths + estimates closing speed, gates outliers, coasts over short dropouts).
    4. Both tracks are extrapolated to the same instant (sensors fire 40 ms apart).
    5. Fusion: if the two ranges are geometrically consistent, triangulate
       (circle intersection) to get x/y/bearing; otherwise fall back gracefully.
    6. Hazard level from range AND time-to-contact.

  OUTPUT (one JSON object per line, ~12 Hz). Distances in cm, bearing in degrees
  (negative = left, positive = right), vel in cm/s (negative = closing).
    {"seq":42,"t":51234,"l":142.3,"r":150.1,"rng":145.0,"brg":-3.2,"x":-6.1,"y":144.8,
     "vel":-35.2,"ttc":4.1,"mode":"tri","zone":"center","haz":"caution","conf":1.0}

  INPUT (optional, from Pi): "T22.5\n" sets air temperature in C for speed-of-sound
  compensation.
*/

#include <Arduino.h>
#include <math.h>
#include "esp_timer.h"
#include "driver/gpio.h"

// ====================== CONFIG ======================
constexpr uint8_t  TRIG_L = 25, ECHO_L = 34;
constexpr uint8_t  TRIG_R = 26, ECHO_R = 35;

constexpr float    BASELINE_CM = 14.0f;   // distance between the two sensor centres - MEASURE YOURS
constexpr float    BASE_TOL_CM = 4.0f;    // slack on the triangle-inequality check (noise)

constexpr uint32_t SLOT_MS     = 40;      // time per sensor (>= 2*maxRange/c + ring-down); 2 sensors -> ~12.5 Hz
constexpr float    MIN_CM      = 2.0f;    // HC-SR04 blind zone
constexpr float    MAX_CM      = 300.0f;  // usable range for small obstacles (spec says 400)

constexpr float    ALPHA       = 0.5f;    // alpha-beta position gain
constexpr float    BETA        = 0.15f;   // alpha-beta velocity gain
constexpr float    GATE_CM     = 60.0f;   // innovation gate; persistent jumps (3 frames) reset the track
constexpr float    V_MAX_CMS   = 400.0f;  // clamp on estimated closing speed
constexpr uint8_t  MAX_COAST   = 2;       // frames to coast on after echoes disappear

constexpr float    DANGER_CM   = 100.0f;
constexpr float    CAUTION_CM  = 200.0f;
constexpr float    AWARE_CM    = 300.0f;
constexpr float    TTC_DANGER_S  = 1.5f;
constexpr float    TTC_CAUTION_S = 3.0f;
constexpr float    ZONE_DEG    = 10.0f;   // |bearing| beyond this = left/right instead of center

float gTempC = 20.0f;                     // updated by "T<celsius>" command

// ====================== SENSOR STATE ======================
struct Sensor {
  uint8_t trig, echo;
  volatile uint32_t tRise;
  volatile uint32_t tFall;
  volatile bool gotFall;

  float   win[3];          // last 3 raw readings (NAN = no echo)
  uint8_t wi;

  bool     valid;          // tracker has a lock
  float    x;              // filtered range (cm)
  float    v;              // filtered range-rate (cm/s)
  uint32_t tLast;          // ms timestamp of last tracker update
  uint8_t  misses, jumps;

  Sensor(uint8_t t, uint8_t e)
    : trig(t), echo(e), tRise(0), tFall(0), gotFall(false), wi(0),
      valid(false), x(0), v(0), tLast(0), misses(0), jumps(0) {
    win[0] = win[1] = win[2] = NAN;
  }
};

Sensor SL(TRIG_L, ECHO_L);
Sensor SR(TRIG_R, ECHO_R);
Sensor* sensors[2] = { &SL, &SR };

int8_t   activeIdx = -1;
uint32_t slotStart = 0;
uint32_t seqNo = 0;

// ====================== LOW LEVEL ======================
void IRAM_ATTR echoISR(void* arg) {
  Sensor* s = (Sensor*)arg;
  uint32_t now = (uint32_t)esp_timer_get_time();
  if (gpio_get_level((gpio_num_t)s->echo)) {
    s->tRise = now;
    s->gotFall = false;
  } else if (s->tRise) {
    s->tFall = now;
    s->gotFall = true;
  }
}

inline float cmPerUs() {                       // round-trip already halved
  return (331.3f + 0.606f * gTempC) / 20000.0f;
}

void fire(Sensor& s) {
  s.tRise = 0;
  s.gotFall = false;
  digitalWrite(s.trig, LOW);
  delayMicroseconds(2);
  digitalWrite(s.trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(s.trig, LOW);
}

float readRaw(Sensor& s) {
  if (!s.gotFall) return NAN;
  uint32_t pulse = s.tFall - s.tRise;
  float cm = pulse * cmPerUs();
  if (cm < MIN_CM || cm > MAX_CM) return NAN;
  return cm;
}

// ====================== FILTERING ======================
float median3(const float* w) {
  float a[3];
  uint8_t n = 0;
  for (uint8_t i = 0; i < 3; i++) if (!isnan(w[i])) a[n++] = w[i];
  if (n < 2) return NAN;                        // need 2 of last 3 to call it a detection
  if (n == 2) return 0.5f * (a[0] + a[1]);
  if (a[0] > a[1]) { float t = a[0]; a[0] = a[1]; a[1] = t; }
  if (a[1] > a[2]) { float t = a[1]; a[1] = a[2]; a[2] = t; }
  if (a[0] > a[1]) { float t = a[0]; a[0] = a[1]; a[1] = t; }
  return a[1];
}

void updateTrack(Sensor& s, float z, uint32_t now) {
  float dt = s.tLast ? (now - s.tLast) / 1000.0f : 0.08f;
  if (dt < 0.01f) dt = 0.01f;
  if (dt > 0.5f)  dt = 0.5f;

  if (!isnan(z)) {
    if (!s.valid) {
      s.x = z; s.v = 0; s.valid = true; s.jumps = 0;
    } else {
      float xp  = s.x + s.v * dt;               // predict
      float res = z - xp;                       // innovation
      if (fabsf(res) > GATE_CM) {
        if (++s.jumps >= 3) { s.x = z; s.v = 0; s.jumps = 0; }   // new object really is there
        else                { s.x = xp; }                        // ignore as outlier
      } else {
        s.jumps = 0;
        s.x = xp + ALPHA * res;
        s.v += (BETA / dt) * res;
        s.v = constrain(s.v, -V_MAX_CMS, V_MAX_CMS);
      }
    }
    s.misses = 0;
  } else if (s.valid) {
    if (++s.misses > MAX_COAST) { s.valid = false; s.v = 0; }
    else                        { s.x += s.v * dt; }
  }
  if (s.valid && s.x < MIN_CM) s.x = MIN_CM;
  s.tLast = now;
}

// Track extrapolated to "now" so L and R refer to the same instant.
float rangeAt(const Sensor& s, uint32_t now) {
  if (!s.valid) return NAN;
  float r = s.x + s.v * ((now - s.tLast) / 1000.0f);
  return r < MIN_CM ? MIN_CM : r;
}

// ====================== FUSION + OUTPUT ======================
static const char* fmt(char* buf, size_t n, float v, int dec) {
  if (isnan(v)) return "null";
  snprintf(buf, n, "%.*f", dec, v);
  return buf;
}

void fuseAndEmit(uint32_t now) {
  const float half = BASELINE_CM * 0.5f;
  float rL = rangeAt(SL, now), rR = rangeAt(SR, now);
  bool vL = !isnan(rL), vR = !isnan(rR);

  float x = NAN, y = NAN, rng = NAN, brg = NAN, conf = 0.0f;
  const char* mode = "none";
  const char* zone = "none";

  if (vL && vR) {
    if (fabsf(rL - rR) <= BASELINE_CM + BASE_TOL_CM) {
      // Circle intersection: L at (-b/2,0), R at (+b/2,0), +x = right, +y = forward
      x = (rL * rL - rR * rR) / (2.0f * BASELINE_CM);
      float h  = x + half;
      float y2 = rL * rL - h * h;
      y = y2 > 0 ? sqrtf(y2) : 0.0f;
      mode = "tri";
      conf = 1.0f;
    } else {
      // Ranges differ by more than the baseline allows: two different objects
      // (or one specular miss). Report the nearer one on its side.
      bool leftNear = rL < rR;
      y = leftNear ? rL : rR;
      x = leftNear ? -half : half;
      mode = "split";
      conf = 0.6f;
    }
  } else if (vL || vR) {
    y = vL ? rL : rR;
    x = vL ? -half : half;
    mode = vL ? "left_only" : "right_only";
    conf = 0.5f;
  }

  if (!isnan(x)) {
    rng = sqrtf(x * x + y * y);
    brg = atan2f(x, y) * 57.29578f;
    if (SL.valid && SL.misses) conf -= 0.2f;    // coasting = less trust
    if (SR.valid && SR.misses) conf -= 0.2f;
    if (conf < 0.1f) conf = 0.1f;

    if      (!strcmp(mode, "left_only"))  zone = "left";
    else if (!strcmp(mode, "right_only")) zone = "right";
    else if (brg < -ZONE_DEG || (!strcmp(mode, "split") && x < 0)) zone = "left";
    else if (brg >  ZONE_DEG || (!strcmp(mode, "split") && x > 0)) zone = "right";
    else zone = "center";
  }

  // Relative closing speed (includes the wearer's own walking speed)
  float vel = NAN;
  if (vL && vR) vel = 0.5f * (SL.v + SR.v);
  else if (vL)  vel = SL.v;
  else if (vR)  vel = SR.v;

  float ttc = NAN;
  if (!isnan(vel) && !isnan(rng) && vel < -10.0f) ttc = rng / (-vel);

  uint8_t haz = 0;
  if (!isnan(rng)) {
    if      (rng < DANGER_CM)  haz = 3;
    else if (rng < CAUTION_CM) haz = 2;
    else if (rng < AWARE_CM)   haz = 1;
    if (!isnan(ttc) && rng < AWARE_CM) {
      if      (ttc < TTC_DANGER_S)  haz = 3;
      else if (ttc < TTC_CAUTION_S) haz = max<uint8_t>(haz, 2);
    }
  }
  static const char* HAZ[4] = { "clear", "aware", "caution", "danger" };

  char b[8][12];
  char line[256];
  snprintf(line, sizeof(line),
    "{\"seq\":%lu,\"t\":%lu,\"l\":%s,\"r\":%s,\"rng\":%s,\"brg\":%s,\"x\":%s,\"y\":%s,"
    "\"vel\":%s,\"ttc\":%s,\"mode\":\"%s\",\"zone\":\"%s\",\"haz\":\"%s\",\"conf\":%s}",
    (unsigned long)seqNo++, (unsigned long)now,
    fmt(b[0], 12, rL, 1), fmt(b[1], 12, rR, 1), fmt(b[2], 12, rng, 1), fmt(b[3], 12, brg, 1),
    fmt(b[4], 12, x, 1), fmt(b[5], 12, y, 1), fmt(b[6], 12, vel, 0), fmt(b[7], 12, ttc, 1),
    mode, zone, HAZ[haz], isnan(conf) ? "0" : (conf >= 0.95f ? "1.0" : (conf >= 0.75f ? "0.8" : (conf >= 0.55f ? "0.6" : "0.4"))));
  Serial.println(line);
}

// ====================== SERIAL INPUT ======================
void handleSerial() {
  static char buf[24];
  static uint8_t n = 0;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (n > 1 && (buf[0] == 'T' || buf[0] == 't')) {
        buf[n] = 0;
        float t = atof(buf + 1);
        if (t > -30.0f && t < 60.0f) {
          gTempC = t;
          Serial.printf("{\"event\":\"temp\",\"c\":%.1f}\n", gTempC);
        }
      }
      n = 0;
    } else if (n < sizeof(buf) - 1) {
      buf[n++] = c;
    }
  }
}

// ====================== ARDUINO ======================
void setup() {
  Serial.begin(115200);
  for (Sensor* s : sensors) {
    pinMode(s->trig, OUTPUT);
    digitalWrite(s->trig, LOW);
    pinMode(s->echo, INPUT);
    attachInterruptArg(digitalPinToInterrupt(s->echo), echoISR, s, CHANGE);
  }
  delay(100);
  Serial.println("{\"event\":\"boot\",\"fw\":\"glasses_ultrasonic\",\"baseline_cm\":14.0}");
  slotStart = millis();
}

void loop() {
  handleSerial();

  uint32_t now = millis();
  if (now - slotStart < SLOT_MS) return;
  slotStart = now;

  // Finish the sensor that just had its slot
  if (activeIdx >= 0) {
    Sensor& s = *sensors[activeIdx];
    s.win[s.wi] = readRaw(s);
    s.wi = (s.wi + 1) % 3;
    updateTrack(s, median3(s.win), now);
    if (activeIdx == 1) fuseAndEmit(now);       // after R updates, both are fresh -> emit
  }

  // Start the next sensor (alternate L, R, L, R ...)
  activeIdx = (activeIdx + 1) % 2;
  fire(*sensors[activeIdx]);
}
