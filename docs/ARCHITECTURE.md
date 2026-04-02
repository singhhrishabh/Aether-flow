# Aether-Flow — Detailed System Architecture

## Overview

Aether-Flow is a three-layer predictive thermal management system designed for device-level deployment. The layers operate across two microcontrollers communicating over USB serial.

## Layer 1 — Perception Layer (ESP32)

The perception layer runs entirely on the ESP32-WROOM-32 and is responsible for sensor polling and telemetry transmission.

### Sensors

| Sensor | Bus | Address | Polling Rate | Purpose |
|---|---|---|---|---|
| TMP117 ×6 | I²C | 0x48–0x4B | 500 ms | Zone-level temperature (±0.1 °C) |
| MLX90640 | I²C | 0x33 | 500 ms | 32×24 thermal image for heat distribution |
| INA226 ×4 | I²C | 0x40–0x43 | 500 ms | CPU/GPU/RAM/system rail power draw |
| SHT40 | I²C | 0x44 | 500 ms | Humidity + ambient temperature |
| YF-S201 | GPIO (interrupt) | — | Continuous | Coolant flow rate |

### Data Format

The ESP32 transmits sensor readings as compact JSON over USB serial at 115200 baud:

```json
{
  "t": 47.3,    // target temperature (°C)
  "ta": 24.1,   // ambient temperature (°C)
  "h": 52.1,    // humidity (%)
  "dp": 14.2,   // dew point (°C)
  "w": 12.4,    // power draw (W)
  "v": 12.1,    // bus voltage (V)
  "p": 120,     // peltier duty (0–255)
  "f1": 180,    // fan 1 duty (0–255)
  "f2": 180,    // fan 2 duty (0–255)
  "r1": 1200,   // fan 1 RPM
  "r2": 1180,   // fan 2 RPM
  "mode": "PID",// control mode
  "safe": 1     // safety status (1=OK, 0=emergency)
}
```

## Layer 2 — Intelligence Layer (Raspberry Pi 5)

### Data Logger (`AetherFlow_DataLogger.py`)

- Reads JSON telemetry from ESP32 via pyserial
- Saves to CSV (training dataset) and InfluxDB (live dashboard)
- Maintains a 120-sample sliding window (60 seconds) for real-time inference
- Sends AI control commands back to ESP32 when model is loaded

### Predictive Model

- **Architecture:** Temporal Fusion Transformer (TFT) with multi-horizon output
- **Fallback architecture:** LSTM with 2 layers (64 → 32 units) + Dense head
- **Input shape:** `(1, 120, 10)` — batch × time steps × features
- **Output:** Temperature predictions at +5 s, +10 s, +15 s, +20 s, +30 s
- **Deployment:** TFLite INT8 quantised for <30 ms inference

### Training Pipeline (`AetherFlow_Train.py`)

1. Load CSV data → drop NaN → sort by time
2. Select 9 input features and 2 output targets
3. MinMaxScaler normalisation (0–1)
4. Sliding window creation (120-step)
5. 80/20 temporal train/val split (no shuffling)
6. LSTM model training with EarlyStopping and ReduceLROnPlateau
7. Export to TFLite with DEFAULT quantisation

## Layer 3 — Actuation Layer (ESP32)

### Control Outputs

| Actuator | PWM Channel | GPIO Pin | PWM Freq | Control Range |
|---|---|---|---|---|
| TEC1-12706 Peltier ×4 | CH0 | GPIO25 | 25 kHz | 0–255 |
| Noctua Fan 1 | CH1 | GPIO26 | 25 kHz | 40–255 (min prevents stall) |
| Noctua Fan 2 | CH2 | GPIO27 | 25 kHz | 40–255 |

### PID Controller (Fallback)

Active when no AI commands received within 2 seconds.

```
Kp = 2.5    (proportional gain)
Ki = 0.1    (integral gain, anti-windup ±100)
Kd = 1.0    (derivative gain)
Target = 45.0 °C
```

### Safety Subsystems

1. **Thermal cutoff:** T > 80 °C → emergency stop (all PWM → 0, relay open)
2. **Power cutoff:** P > 200 W → emergency stop
3. **Dew-point guard:** T < (dew point + 5 °C) → reduce Peltier duty by 20/tick
4. **AI timeout:** No AI command for 2 s → revert to PID
5. **Hardware relay:** GPIO32, physically disconnects Peltier power rail

## Communication Protocol

```
┌───────────┐          USB Serial (115200)          ┌────────────┐
│  ESP32    │  ─────── JSON telemetry ──────────▶   │  RPi 5     │
│           │  ◀─────  JSON commands  ──────────    │            │
│  500 ms   │                                       │  AI model  │
│  PID loop │                                       │  TFLite    │
└───────────┘                                       └────────────┘
```

### Telemetry (ESP32 → RPi): every 500 ms
```json
{"t":47.3,"ta":24.1,"h":52.1,"dp":14.2,"w":12.4,"v":12.1,"p":120,"f1":180,"f2":180,"r1":1200,"r2":1180,"mode":"PID","safe":1}
```

### Commands (RPi → ESP32): as needed
```json
{"peltier":120,"fan1":180,"fan2":180}
```
