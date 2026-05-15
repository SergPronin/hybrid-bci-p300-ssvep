#include <DueTimer.h>
#include <math.h>

// ================== НАСТРОЙКИ ==================
volatile long freq[6] = {10000, 10000, 10000, 10000, 10000, 10000};
float targetFreq[6] = {0, 0, 0, 0, 0, 0};

int mode = 0; // 0 = постоянный, 1 = пакетный

// Период одного переключения (полупериод) в мкс: 1e6 / (2 * f_hz) = 500000 / f_hz
float period_float = 500000.0;

volatile long mask_1 = 0x7E;
volatile long new_portc = 0;

// ===== СОСТОЯНИЯ =====
volatile bool ledState[6] = {false};
volatile bool ledChanged[6] = {false};

DueTimer led_timers[6] = {
  DueTimer(0), DueTimer(1), DueTimer(2),
  DueTimer(3), DueTimer(4), DueTimer(5)
};

// ===== ПАКЕТНЫЙ РЕЖИМ =====
const uint32_t MIN_BURST_MS = 2000;
const uint32_t MAX_BURST_MS = 2000;

const uint32_t MIN_PAUSE_MS = 2000;
const uint32_t MAX_PAUSE_MS = 3000;

bool isBursting[6] = {false};
uint32_t phaseStartMillis[6] = {0};
uint32_t phaseDurationMillis[6] = {0};

static inline uint32_t ledPinMask(int i) {
  return 1UL << (i + 1);
}

// Синхронная установка пина (и shadow new_portc) — из ISR и из loop
static void setLedPin(int i, bool on) {
  const uint32_t bit = ledPinMask(i);
  if (on) {
    PIOC->PIO_SODR = bit;
    new_portc |= bit;
  } else {
    PIOC->PIO_CODR = bit;
    new_portc &= ~bit;
  }
  ledState[i] = on;
}

static void lampOff(int i) {
  led_timers[i].stop();
  setLedPin(i, false);
}

static bool freqChanged(int i, float newHz) {
  return fabs(targetFreq[i] - newHz) > 0.001f;
}

static void beginBurstPhase(int i) {
  isBursting[i] = true;
  phaseStartMillis[i] = millis();
  phaseDurationMillis[i] = random(MIN_BURST_MS, MAX_BURST_MS + 1);
  setLedPin(i, false);
  led_timers[i].stop();
  led_timers[i].start(freq[i]);
}

static void beginPausePhase(int i) {
  isBursting[i] = false;
  phaseStartMillis[i] = millis();
  phaseDurationMillis[i] = random(MIN_PAUSE_MS, MAX_PAUSE_MS + 1);
  lampOff(i);
}

// Применить частоту с учётом режима (force — перезапуск даже если Hz тот же)
static void applyTargetFreq(int i, float temp_freq, bool force) {
  if (!force && !freqChanged(i, temp_freq)) {
    return;
  }

  targetFreq[i] = temp_freq;

  if (temp_freq > 0.0f) {
    freq[i] = (long)lround(period_float / temp_freq);

    if (mode == 0) {
      setLedPin(i, false);
      led_timers[i].stop();
      led_timers[i].start(freq[i]);
    } else {
      beginBurstPhase(i);
    }
  } else {
    lampOff(i);
    isBursting[i] = false;
    targetFreq[i] = 0.0f;
  }
}

// ================== TIMER ==================
void Timer0_action() { bool on = !ledState[0]; setLedPin(0, on); ledChanged[0] = true; }
void Timer1_action() { bool on = !ledState[1]; setLedPin(1, on); ledChanged[1] = true; }
void Timer2_action() { bool on = !ledState[2]; setLedPin(2, on); ledChanged[2] = true; }
void Timer3_action() { bool on = !ledState[3]; setLedPin(3, on); ledChanged[3] = true; }
void Timer4_action() { bool on = !ledState[4]; setLedPin(4, on); ledChanged[4] = true; }
void Timer5_action() { bool on = !ledState[5]; setLedPin(5, on); ledChanged[5] = true; }

// ================== PARSER ==================
void handleCommand(String cmd) {
  cmd.trim();

  if (cmd.startsWith("M")) {
    int newMode = cmd.substring(2).toInt();
    mode = newMode;
    // После смены режима перезапускаем все активные лампы (как в WinForms после M + частот)
    for (int i = 0; i < 6; i++) {
      if (targetFreq[i] > 0.0f) {
        applyTargetFreq(i, targetFreq[i], true);
      }
    }
    Serial.print("Mode=");
    Serial.println(mode);
    return;
  }

  int start = 0;
  for (int i = 0; i < 6; i++) {
    int space = cmd.indexOf(' ', start);
    String part = (space == -1) ? cmd.substring(start) : cmd.substring(start, space);
    float temp_freq = part.toFloat();
    applyTargetFreq(i, temp_freq, false);

    if (space == -1) {
      break;
    }
    start = space + 1;
  }
}

// ================== SETUP ==================
void setup() {
  PIOC->PIO_OER = mask_1;
  PIOC->PIO_CODR = mask_1;
  new_portc = 0;

  led_timers[0].attachInterrupt(Timer0_action);
  led_timers[1].attachInterrupt(Timer1_action);
  led_timers[2].attachInterrupt(Timer2_action);
  led_timers[3].attachInterrupt(Timer3_action);
  led_timers[4].attachInterrupt(Timer4_action);
  led_timers[5].attachInterrupt(Timer5_action);

  Serial.begin(115200);
  Serial.setTimeout(50);

  randomSeed(analogRead(0));
}

// ================== LOOP ==================
void loop() {
  // События для Python (не блокируем ISR — пины уже выставлены в прерывании)
  for (int i = 0; i < 6; i++) {
    if (ledChanged[i]) {
      ledChanged[i] = false;
      Serial.print("LED ");
      Serial.print(i);
      Serial.print(" ");
      Serial.print(millis());
      Serial.print(" ");
      Serial.println(ledState[i] ? "ON" : "OFF");
    }
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  if (mode == 1) {
    uint32_t now = millis();

    for (int i = 0; i < 6; i++) {
      if (targetFreq[i] <= 0.0f) {
        continue;
      }

      if (now - phaseStartMillis[i] >= phaseDurationMillis[i]) {
        if (isBursting[i]) {
          beginPausePhase(i);
        } else {
          beginBurstPhase(i);
        }
      }
    }
  }
}
