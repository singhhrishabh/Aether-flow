# Aether-Flow — Firmware Deployment Guide

## Overview

This guide covers flashing the `AetherFlow_Firmware.ino` to an ESP32-WROOM-32 using the Arduino IDE.

## Prerequisites

- **Arduino IDE 2.x** (download from [arduino.cc](https://www.arduino.cc/en/software))
- **ESP32 board support** installed in Arduino IDE
- **USB cable** (data-capable, not charge-only)

## Step 1 — Install ESP32 Board Support

1. Open Arduino IDE → **File → Preferences**
2. In "Additional Board Manager URLs", add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools → Board → Boards Manager**
4. Search for "ESP32" → Install **esp32 by Espressif Systems**

## Step 2 — Install Required Libraries

Go to **Tools → Manage Libraries** and install:

| Library | Author | Purpose |
|---|---|---|
| Adafruit TMP117 | Adafruit | Temperature sensor driver |
| Adafruit SHT4x | Adafruit | Humidity sensor driver |
| INA226_WE | Wolfgang Ewald | Power monitor driver |
| ArduinoJson | Benoit Blanchon | JSON serialisation for telemetry |

## Step 3 — Wiring

| Sensor / Device | ESP32 Pin | Notes |
|---|---|---|
| TMP117 SDA | GPIO21 | I²C Data (shared bus) |
| TMP117 SCL | GPIO22 | I²C Clock (shared bus) |
| SHT40 SDA | GPIO21 | Same I²C bus |
| SHT40 SCL | GPIO22 | Same I²C bus |
| INA226 SDA | GPIO21 | Same I²C bus |
| INA226 SCL | GPIO22 | Same I²C bus |
| Peltier MOSFET gate | GPIO25 | PWM channel 0 |
| Fan 1 PWM | GPIO26 | PWM channel 1 |
| Fan 2 PWM | GPIO27 | PWM channel 2 |
| Fan 1 Tach | GPIO34 | Input only (interrupt) |
| Fan 2 Tach | GPIO35 | Input only (interrupt) |
| Safety relay | GPIO32 | HIGH = cooling ON |
| Status LED | GPIO2 | Built-in LED |
| All sensors VCC | 3.3 V | — |
| All sensors GND | GND | — |

> **Important:** All I²C sensors share the same SDA/SCL bus. Each sensor has a unique I²C address so there are no conflicts.

## Step 4 — Flash Firmware

1. Connect ESP32 to your computer via USB
2. Open `firmware/AetherFlow_Firmware.ino` in Arduino IDE
3. Select board: **Tools → Board → ESP32 Dev Module**
4. Select port: **Tools → Port → /dev/ttyUSB0** (or COM3 on Windows)
5. Click **Upload** (→ button)
6. Wait for "Done uploading" message
7. The status LED should blink 5 times to confirm successful boot

## Step 5 — Verify

1. Open **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. You should see:
   ```
   ===========================================
     AETHER-FLOW FIRMWARE v1.0 STARTING...
   ===========================================
   [I2C] Bus initialized on GPIO21/22
   [OK] TMP117 temperature sensor ready
   [OK] SHT40 humidity sensor ready
   [OK] INA226 power monitor ready
   [READY] Aether-Flow system initialized.
   ```
4. JSON telemetry should appear every 500 ms

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "TMP117 not found" | Wiring issue or wrong I²C address | Check SDA/SCL connections, run I²C scanner |
| No serial output | Wrong baud rate | Set Serial Monitor to 115200 |
| Upload fails | Wrong board or port | Verify ESP32 Dev Module selected |
| LED not blinking | Upload didn't complete | Hold BOOT button during upload |
| Random characters | Baud rate mismatch | Ensure 115200 on both ends |

## PID Tuning

The firmware ships with default PID gains:

```
Kp = 2.5    (increase if response too slow)
Ki = 0.1    (increase if steady-state error persists)
Kd = 1.0    (increase if oscillating/overshooting)
```

To tune, modify the `#define` values near the top of `AetherFlow_Firmware.ino` and re-upload.
