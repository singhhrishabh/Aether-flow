// ============================================================
//  AETHER-FLOW FIRMWARE v1.0
//  AI-Driven Molecular Cooling Infrastructure
//  Target Hardware: ESP32-WROOM-32
// ============================================================
//
//  WHAT THIS CODE DOES (Beginner Explanation):
//  --------------------------------------------
//  Think of this firmware as the "nervous system" of Aether-Flow.
//  It does 4 things in a loop, forever:
//
//  1. READ   → Collects temperature, power, and humidity data
//              from all sensors every 500ms
//
//  2. THINK  → Runs a simple control algorithm (PID controller)
//              that decides HOW MUCH cooling is needed right now
//
//  3. ACT    → Sends PWM signals to the Peltier modules and fans
//              to deliver exactly that amount of cooling
//
//  4. REPORT → Sends all data over Serial (USB) to the Raspberry Pi
//              where the AI model lives
//
//  The Raspberry Pi reads this data, runs the AI prediction,
//  and sends back smarter control commands. The ESP32 always
//  has the PID as a FALLBACK if the Pi goes silent.
//
// ============================================================
//
//  WIRING GUIDE:
//  ─────────────────────────────────────────────────────────
//  SENSOR / DEVICE          ESP32 PIN    NOTES
//  ─────────────────────────────────────────────────────────
//  TMP117 (Temp Sensor)     SDA → GPIO21  I2C Data
//                           SCL → GPIO22  I2C Clock
//                           VCC → 3.3V
//                           GND → GND
//
//  SHT40 (Humidity Sensor)  SDA → GPIO21  Same I2C bus
//                           SCL → GPIO22
//                           VCC → 3.3V
//                           GND → GND
//
//  INA226 (Power Monitor)   SDA → GPIO21  Same I2C bus
//                           SCL → GPIO22
//                           VCC → 3.3V
//                           GND → GND
//
//  Peltier PWM (via MOSFET) GPIO25        PWM signal (0-255)
//  Fan 1 PWM                GPIO26        PWM signal (0-255)
//  Fan 2 PWM                GPIO27        PWM signal (0-255)
//  Fan 1 Tach (RPM)         GPIO34        Input only
//  Fan 2 Tach (RPM)         GPIO35        Input only
//
//  Safety Cutoff Relay      GPIO32        HIGH = cooling ON
//                                         LOW  = emergency OFF
//
//  Status LED (built-in)    GPIO2         Blinks to show activity
//
// ============================================================
//
//  LIBRARIES YOU NEED TO INSTALL:
//  Open Arduino IDE → Tools → Manage Libraries → Search & Install:
//  1. "Adafruit TMP117"    by Adafruit
//  2. "Adafruit SHT4x"     by Adafruit
//  3. "INA226_WE"          by Wolfgang Ewald
//  4. "ArduinoJson"        by Benoit Blanchon
//
// ============================================================

#include <Wire.h>              // I2C communication (built-in)
#include <Arduino.h>           // Core Arduino functions (built-in)
#include <Adafruit_TMP117.h>   // TMP117 temperature sensor
#include <Adafruit_SHT4x.h>    // SHT40 humidity sensor
#include <INA226_WE.h>         // INA226 power monitor
#include <ArduinoJson.h>       // JSON formatting for data sending


// ============================================================
//  PIN DEFINITIONS
//  (These numbers tell the ESP32 which physical pin to use)
// ============================================================
#define PIN_PELTIER_PWM   25   // Controls Peltier cooling power (0 = off, 255 = full)
#define PIN_FAN1_PWM      26   // Controls Fan 1 speed
#define PIN_FAN2_PWM      27   // Controls Fan 2 speed
#define PIN_FAN1_TACH     34   // Reads Fan 1 RPM (how fast it's spinning)
#define PIN_FAN2_TACH     35   // Reads Fan 2 RPM
#define PIN_SAFETY_RELAY  32   // Emergency shutoff switch
#define PIN_STATUS_LED    2    // Built-in LED on ESP32

// PWM settings
// PWM = Pulse Width Modulation — a way to control power by
// switching it on/off very fast. Think of it like a light dimmer.
#define PWM_FREQ          25000  // 25kHz — too fast for humans to hear (avoids whining noise)
#define PWM_RESOLUTION    8      // 8-bit = values from 0 to 255
#define PWM_CH_PELTIER    0      // PWM channel 0 for Peltier
#define PWM_CH_FAN1       1      // PWM channel 1 for Fan 1
#define PWM_CH_FAN2       2      // PWM channel 2 for Fan 2


// ============================================================
//  SAFETY LIMITS
//  If any of these are exceeded, the system shuts down safely
// ============================================================
#define TEMP_MAX_SAFE       85.0   // °C — above this = emergency shutdown
#define TEMP_TARGET         45.0   // °C — this is what we're trying to maintain
#define TEMP_DEWPOINT_GUARD  5.0   // °C — Peltier won't cool below (dew point + this buffer)
                                   // (prevents condensation = water damage)
#define POWER_MAX_WATTS     60.0   // Watts — max power draw allowed
#define AI_TIMEOUT_MS       2000   // If no AI command in 2 seconds, use PID fallback


// ============================================================
//  PID CONTROLLER SETTINGS
//
//  PID = Proportional, Integral, Derivative
//  It's a classic control algorithm. Think of it like a human
//  adjusting a shower temperature:
//
//  P (Proportional) = "How far am I from the target right now?"
//                     → Big gap = big correction
//
//  I (Integral)     = "Have I been too hot for a long time?"
//                     → Accumulated error = extra correction
//
//  D (Derivative)   = "Is the temperature changing fast?"
//                     → Rising fast = slow down before overshooting
//
//  These three numbers (Kp, Ki, Kd) tune how aggressive the
//  controller is. Start with these values and adjust if needed.
// ============================================================
#define PID_KP   2.5    // Proportional gain — main correction strength
#define PID_KI   0.08   // Integral gain — fixes slow drift
#define PID_KD   1.2    // Derivative gain — prevents overshoot


// ============================================================
//  SENSOR OBJECTS
//  These are like "handles" to talk to each sensor
// ============================================================
Adafruit_TMP117  tmp117;          // Temperature sensor
Adafruit_SHT4x   sht40;          // Humidity sensor
INA226_WE        ina226(0x40);    // Power monitor (I2C address 0x40)


// ============================================================
//  GLOBAL STATE VARIABLES
//  These variables store the current state of the system
// ============================================================

// --- Sensor readings ---
float tempCelsius       = 25.0;   // Current temperature in °C
float tempAmbient       = 25.0;   // Room temperature in °C
float humidityPercent   = 50.0;   // Relative humidity %
float dewPointCelsius   = 15.0;   // Calculated dew point °C
float powerWatts        = 0.0;    // Current power draw in Watts
float voltsBus          = 0.0;    // Bus voltage in Volts
float fan1RPM           = 0.0;    // Fan 1 speed (rotations per minute)
float fan2RPM           = 0.0;    // Fan 2 speed

// --- Control outputs (0-255) ---
uint8_t peltierDuty     = 0;      // How hard the Peltier is working (0=off, 255=max)
uint8_t fan1Duty        = 0;      // Fan 1 speed
uint8_t fan2Duty        = 0;      // Fan 2 speed

// --- PID controller state ---
float pidError          = 0.0;    // Current error (target - actual temp)
float pidLastError      = 0.0;    // Previous error (for derivative)
float pidIntegral       = 0.0;    // Accumulated error (for integral)
float pidOutput         = 0.0;    // PID output (becomes duty cycle)

// --- System state ---
bool  safetyShutdown    = false;  // True = emergency stop triggered
bool  aiControlActive   = false;  // True = AI is sending commands
bool  systemReady       = false;  // True = all sensors initialized OK
unsigned long lastAiMsg = 0;      // Timestamp of last AI command received
unsigned long lastLoop  = 0;      // Timestamp of last main loop

// --- Fan tach counting (interrupt-based) ---
volatile int fan1Pulses = 0;
volatile int fan2Pulses = 0;


// ============================================================
//  INTERRUPT SERVICE ROUTINES
//
//  "Interrupt" = a special function that runs IMMEDIATELY when
//  a pin changes state, interrupting whatever else is running.
//  We use this to count fan pulses accurately.
//  Each fan rotation generates 2 pulses — we count them.
// ============================================================
void IRAM_ATTR fan1TachISR() { fan1Pulses++; }
void IRAM_ATTR fan2TachISR() { fan2Pulses++; }


// ============================================================
//  SETUP — Runs ONCE when the ESP32 powers on
// ============================================================
void setup() {

  // Start serial communication at 115200 baud
  // This lets us print messages to the computer (and receive AI commands)
  Serial.begin(115200);
  delay(500);

  Serial.println("===========================================");
  Serial.println("  AETHER-FLOW FIRMWARE v1.0 STARTING...");
  Serial.println("===========================================");

  // --- Set up pins ---
  pinMode(PIN_SAFETY_RELAY, OUTPUT);
  pinMode(PIN_STATUS_LED,   OUTPUT);
  pinMode(PIN_FAN1_TACH,    INPUT_PULLUP);  // Tach pins are inputs
  pinMode(PIN_FAN2_TACH,    INPUT_PULLUP);

  // --- Safety relay ON (allows cooling) ---
  digitalWrite(PIN_SAFETY_RELAY, HIGH);

  // --- Set up PWM channels ---
  // ledcSetup(channel, frequency, resolution)
  ledcSetup(PWM_CH_PELTIER, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PWM_CH_FAN1,    PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PWM_CH_FAN2,    PWM_FREQ, PWM_RESOLUTION);

  // ledcAttachPin(pin, channel) — link the pin to the PWM channel
  ledcAttachPin(PIN_PELTIER_PWM, PWM_CH_PELTIER);
  ledcAttachPin(PIN_FAN1_PWM,    PWM_CH_FAN1);
  ledcAttachPin(PIN_FAN2_PWM,    PWM_CH_FAN2);

  // Start with everything OFF (safe default)
  ledcWrite(PWM_CH_PELTIER, 0);
  ledcWrite(PWM_CH_FAN1,    0);
  ledcWrite(PWM_CH_FAN2,    0);

  // --- Attach fan tach interrupts ---
  // FALLING = trigger when pulse goes from HIGH to LOW
  attachInterrupt(digitalPinToInterrupt(PIN_FAN1_TACH), fan1TachISR, FALLING);
  attachInterrupt(digitalPinToInterrupt(PIN_FAN2_TACH), fan2TachISR, FALLING);

  // --- Initialize I2C bus ---
  Wire.begin(21, 22);   // SDA = GPIO21, SCL = GPIO22
  Serial.println("[I2C] Bus initialized on GPIO21/22");

  // --- Initialize TMP117 temperature sensor ---
  if (!tmp117.begin()) {
    Serial.println("[ERROR] TMP117 not found! Check wiring.");
    blinkError(3);   // 3 blinks = sensor error
  } else {
    Serial.println("[OK] TMP117 temperature sensor ready");
  }

  // --- Initialize SHT40 humidity sensor ---
  if (!sht40.begin()) {
    Serial.println("[ERROR] SHT40 not found! Check wiring.");
    blinkError(3);
  } else {
    sht40.setPrecision(SHT4X_HIGH_PRECISION);
    Serial.println("[OK] SHT40 humidity sensor ready");
  }

  // --- Initialize INA226 power monitor ---
  if (!ina226.init()) {
    Serial.println("[ERROR] INA226 not found! Check wiring.");
    blinkError(3);
  } else {
    // Configure for a 0.1 ohm shunt resistor
    // The shunt resistor is a tiny precision resistor that lets us
    // measure current by measuring the tiny voltage across it
    ina226.setResistorRange(0.1, 1.3);   // 0.1 ohm, max 1.3A
    Serial.println("[OK] INA226 power monitor ready");
  }

  // --- All done! ---
  systemReady = true;
  Serial.println("[READY] Aether-Flow system initialized.");
  Serial.println("[INFO] Waiting for AI connection from Raspberry Pi...");
  Serial.println("[INFO] Using PID fallback control until AI connects.");

  // Flash LED 5 times to signal successful startup
  for (int i = 0; i < 5; i++) {
    digitalWrite(PIN_STATUS_LED, HIGH);
    delay(100);
    digitalWrite(PIN_STATUS_LED, LOW);
    delay(100);
  }
}


// ============================================================
//  MAIN LOOP — Runs FOREVER, repeating every ~500ms
// ============================================================
void loop() {

  unsigned long now = millis();   // Current time in milliseconds

  // Only run the main loop every 500ms
  // (millis() counts up since boot, so this is a non-blocking timer)
  if (now - lastLoop < 500) return;
  lastLoop = now;

  // Blink LED to show the system is alive
  digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));

  // --- STEP 1: Read all sensors ---
  readSensors();

  // --- STEP 2: Check safety limits ---
  checkSafety();

  // If safety shutdown triggered, everything stops here
  if (safetyShutdown) {
    emergencyStop();
    return;
  }

  // --- STEP 3: Check for incoming AI commands ---
  if (Serial.available()) {
    readAICommand();
  }

  // --- STEP 4: Determine if AI is still active ---
  // If no AI message in the last 2 seconds, fall back to PID
  aiControlActive = (now - lastAiMsg < AI_TIMEOUT_MS) && (lastAiMsg > 0);

  // --- STEP 5: Calculate control output ---
  if (!aiControlActive) {
    // PID FALLBACK — calculate cooling ourselves
    runPIDController();
    applyPIDOutput();
  }
  // If AI is active, outputs were already set in readAICommand()

  // --- STEP 6: Apply outputs to hardware ---
  ledcWrite(PWM_CH_PELTIER, peltierDuty);
  ledcWrite(PWM_CH_FAN1,    fan1Duty);
  ledcWrite(PWM_CH_FAN2,    fan2Duty);

  // --- STEP 7: Report data back to Raspberry Pi (and Serial Monitor) ---
  sendTelemetry();

  // --- STEP 8: Calculate fan RPM from pulse count ---
  // (We sample pulses over 500ms, then calculate RPM)
  // RPM = (pulses / 2 pulses_per_rev) * (60s / 0.5s)
  fan1RPM = (fan1Pulses / 2.0) * 120.0;
  fan2RPM = (fan2Pulses / 2.0) * 120.0;
  fan1Pulses = 0;   // Reset counters for next 500ms window
  fan2Pulses = 0;
}


// ============================================================
//  FUNCTION: readSensors()
//  Reads all sensor values and stores them in global variables
// ============================================================
void readSensors() {

  // --- Read TMP117 temperature ---
  sensors_event_t tempEvent;
  if (tmp117.getEvent(&tempEvent)) {
    tempCelsius = tempEvent.temperature;
  } else {
    Serial.println("[WARN] TMP117 read failed");
  }

  // --- Read SHT40 humidity and ambient temperature ---
  sensors_event_t humEvent, ambEvent;
  if (sht40.getEvent(&humEvent, &ambEvent)) {
    humidityPercent = humEvent.relative_humidity;
    tempAmbient     = ambEvent.temperature;

    // Calculate dew point using the Magnus formula
    // (a mathematical approximation — accurate within 1°C)
    // This tells us: "at what temperature will water start condensing?"
    float a = 17.27;
    float b = 237.7;
    float alpha = ((a * tempCelsius) / (b + tempCelsius)) + log(humidityPercent / 100.0);
    dewPointCelsius = (b * alpha) / (a - alpha);

  } else {
    Serial.println("[WARN] SHT40 read failed");
  }

  // --- Read INA226 power monitor ---
  voltsBus   = ina226.getBusVoltage_V();
  powerWatts = ina226.getBusPower();
}


// ============================================================
//  FUNCTION: checkSafety()
//  Checks all safety conditions and triggers emergency stop if needed
// ============================================================
void checkSafety() {

  // --- Check 1: Is temperature dangerously high? ---
  if (tempCelsius > TEMP_MAX_SAFE) {
    Serial.print("[SAFETY] CRITICAL: Temperature ");
    Serial.print(tempCelsius);
    Serial.println("°C exceeds safe limit! Emergency stop.");
    safetyShutdown = true;
    return;
  }

  // --- Check 2: Is power draw too high? ---
  if (powerWatts > POWER_MAX_WATTS) {
    Serial.print("[SAFETY] CRITICAL: Power draw ");
    Serial.print(powerWatts);
    Serial.println("W exceeds limit! Emergency stop.");
    safetyShutdown = true;
    return;
  }

  // --- Check 3: Dew point guard ---
  // Prevent the Peltier from cooling below the dew point
  // (which would cause water to condense on the electronics)
  float minSafeTemp = dewPointCelsius + TEMP_DEWPOINT_GUARD;
  if (tempCelsius < minSafeTemp && peltierDuty > 0) {
    Serial.print("[SAFETY] Dew point guard active. Min safe temp: ");
    Serial.print(minSafeTemp);
    Serial.println("°C. Reducing Peltier duty.");
    // Reduce Peltier power — don't go below dew point
    peltierDuty = max(0, peltierDuty - 20);
  }
}


// ============================================================
//  FUNCTION: runPIDController()
//  Calculates how much cooling is needed using PID algorithm
//
//  The PID controller works like a smart thermostat:
//  - It knows the TARGET temperature (TEMP_TARGET = 45°C)
//  - It measures the ACTUAL temperature
//  - It calculates the ERROR (how far off we are)
//  - It computes a correction that drives the error to zero
// ============================================================
void runPIDController() {

  // Calculate error: positive = too hot, negative = too cold
  pidError = tempCelsius - TEMP_TARGET;

  // Proportional term: direct correction proportional to error
  // If 10°C too hot → P correction = 10 * 2.5 = 25
  float P = PID_KP * pidError;

  // Integral term: accumulates error over time
  // Fixes "steady-state error" — when PID gets close but never quite reaches target
  pidIntegral += pidError * 0.5;           // 0.5 = loop interval in seconds
  pidIntegral = constrain(pidIntegral, -100, 100);  // Anti-windup: prevent runaway
  float I = PID_KI * pidIntegral;

  // Derivative term: reacts to rate of change
  // Prevents overshoot by anticipating where temperature is heading
  float derivative = (pidError - pidLastError) / 0.5;  // Rate of change per second
  float D = PID_KD * derivative;

  // Total PID output
  pidOutput = P + I + D;
  pidLastError = pidError;
}


// ============================================================
//  FUNCTION: applyPIDOutput()
//  Converts PID output to actual PWM values for Peltier and fans
// ============================================================
void applyPIDOutput() {

  // Only cool if temperature is above target
  if (pidOutput > 0) {

    // Map PID output to Peltier duty cycle (0-255)
    // constrain() makes sure we never go below 0 or above 255
    peltierDuty = (uint8_t) constrain(pidOutput * 3.5, 0, 255);

    // Fan speed follows Peltier power — proportional
    // When Peltier works hard, fans help dissipate more heat
    uint8_t fanSpeed = (uint8_t) constrain(pidOutput * 2.8, 50, 255);
    // Note: fans have a minimum of 50 (below this they may stall)
    fan1Duty = fanSpeed;
    fan2Duty = fanSpeed;

  } else {
    // Temperature is at or below target — reduce cooling
    peltierDuty = 0;

    // Keep fans at minimum speed for airflow
    fan1Duty = 30;
    fan2Duty = 30;
  }
}


// ============================================================
//  FUNCTION: readAICommand()
//  Reads a JSON command from the Raspberry Pi over Serial
//
//  The Raspberry Pi sends commands like this:
//  {"peltier":120,"fan1":180,"fan2":180,"mode":"AI"}
//
//  JSON = JavaScript Object Notation — a simple data format
//  that looks like: { "key": value, "key2": value2 }
// ============================================================
void readAICommand() {

  // Read the full line from Serial
  String line = Serial.readStringUntil('\n');
  line.trim();

  // Skip empty lines or debug messages
  if (line.length() < 5) return;
  if (!line.startsWith("{")) return;

  // Parse JSON
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, line);

  if (err) {
    Serial.print("[WARN] JSON parse error: ");
    Serial.println(err.c_str());
    return;
  }

  // Extract values from JSON (with safe defaults)
  uint8_t newPeltier = doc["peltier"] | peltierDuty;   // | = "or default"
  uint8_t newFan1    = doc["fan1"]    | fan1Duty;
  uint8_t newFan2    = doc["fan2"]    | fan2Duty;

  // Validate ranges before applying
  peltierDuty = constrain(newPeltier, 0, 255);
  fan1Duty    = constrain(newFan1,    0, 255);
  fan2Duty    = constrain(newFan2,    0, 255);

  // Update timestamp — this tells the system AI is still alive
  lastAiMsg = millis();

  Serial.print("[AI CMD] Peltier:");
  Serial.print(peltierDuty);
  Serial.print(" Fan1:");
  Serial.print(fan1Duty);
  Serial.print(" Fan2:");
  Serial.println(fan2Duty);
}


// ============================================================
//  FUNCTION: sendTelemetry()
//  Sends all sensor data as JSON to the Raspberry Pi
//
//  This runs every 500ms. The Pi receives it, feeds it to the
//  AI model, and sends back control commands.
//
//  Output example:
//  {"t":47.3,"ta":24.1,"h":52.1,"dp":14.2,"w":12.4,
//   "v":12.1,"p":120,"f1":180,"f2":180,"r1":1200,"r2":1180,
//   "mode":"PID","safe":1}
// ============================================================
void sendTelemetry() {

  StaticJsonDocument<512> doc;

  doc["t"]    = round(tempCelsius    * 10) / 10.0;   // Target temperature
  doc["ta"]   = round(tempAmbient    * 10) / 10.0;   // Ambient temperature
  doc["h"]    = round(humidityPercent* 10) / 10.0;   // Humidity
  doc["dp"]   = round(dewPointCelsius* 10) / 10.0;   // Dew point
  doc["w"]    = round(powerWatts     * 10) / 10.0;   // Power (watts)
  doc["v"]    = round(voltsBus       * 10) / 10.0;   // Voltage
  doc["p"]    = peltierDuty;                          // Peltier duty (0-255)
  doc["f1"]   = fan1Duty;                             // Fan 1 duty
  doc["f2"]   = fan2Duty;                             // Fan 2 duty
  doc["r1"]   = (int)fan1RPM;                         // Fan 1 RPM
  doc["r2"]   = (int)fan2RPM;                         // Fan 2 RPM
  doc["mode"] = aiControlActive ? "AI" : "PID";       // Control mode
  doc["safe"] = safetyShutdown ? 0 : 1;               // 1 = healthy

  // Serialize and print to Serial (which the Pi reads)
  serializeJson(doc, Serial);
  Serial.println();   // Newline marks end of message
}


// ============================================================
//  FUNCTION: emergencyStop()
//  Shuts everything down safely when a critical condition occurs
// ============================================================
void emergencyStop() {

  // Cut power to Peltier and fans
  ledcWrite(PWM_CH_PELTIER, 0);
  ledcWrite(PWM_CH_FAN1,    0);
  ledcWrite(PWM_CH_FAN2,    0);

  // Open safety relay (physically cuts power to Peltier circuit)
  digitalWrite(PIN_SAFETY_RELAY, LOW);

  // Rapid LED blinking to signal emergency
  // This loops forever — manual reset required to restart
  Serial.println("[EMERGENCY] System halted. Manual reset required.");
  while (true) {
    digitalWrite(PIN_STATUS_LED, HIGH);
    delay(200);
    digitalWrite(PIN_STATUS_LED, LOW);
    delay(200);
  }
}


// ============================================================
//  FUNCTION: blinkError(int times)
//  Blinks the status LED to signal an error during startup
// ============================================================
void blinkError(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(PIN_STATUS_LED, HIGH);
    delay(300);
    digitalWrite(PIN_STATUS_LED, LOW);
    delay(300);
  }
  delay(600);
}


// ============================================================
//  END OF AETHER-FLOW FIRMWARE v1.0
//
//  NEXT STEPS:
//  1. Flash this to ESP32 using Arduino IDE
//  2. Open Serial Monitor at 115200 baud to see live data
//  3. Connect Raspberry Pi to ESP32 via USB
//  4. Run the Python AI script on the Pi (AetherFlow_AI.py)
//
//  The Pi reads JSON from Serial, runs the LSTM model,
//  and writes JSON commands back — completing the AI loop.
// ============================================================
