#!/usr/bin/env python3
"""
============================================================
  AETHER-FLOW  —  Raspberry Pi AI Data Logger
  File: AetherFlow_DataLogger.py
============================================================

WHAT THIS SCRIPT DOES (Beginner Explanation):
----------------------------------------------
This script runs on the Raspberry Pi and does 3 jobs:

  JOB 1 — LISTEN
    Reads the JSON data that the ESP32 sends every 500ms
    over the USB cable (Serial port).

  JOB 2 — SAVE
    Saves every reading to a CSV file on the Pi's SD card.
    This builds up your training dataset over days/weeks.
    Also saves to an InfluxDB database for the live dashboard.

  JOB 3 — CONTROL (Phase 2 — after AI is trained)
    Once you have a trained model, this script loads it and
    sends AI control commands back to the ESP32.
    Until then, it runs in "observation mode" — just watching.

HOW TO RUN:
-----------
  Step 1: Install dependencies (one time only):
    pip3 install pyserial influxdb-client pandas numpy

  Step 2: Find your ESP32's port:
    ls /dev/tty*      (look for /dev/ttyUSB0 or /dev/ttyACM0)

  Step 3: Run this script:
    python3 AetherFlow_DataLogger.py

  Step 4: Watch the terminal — you'll see live readings!
    Press Ctrl+C to stop.

============================================================
"""

from __future__ import annotations

import serial          # Reads data from USB serial port
import json            # Parses JSON strings into Python dicts
import csv             # Saves data to CSV spreadsheet files
import time            # Time functions (sleep, timestamps)
import os              # File and folder operations
import sys             # System functions (exit, etc.)
import threading       # Runs multiple tasks at the same time
from datetime import datetime   # Date and time formatting
from collections import deque   # A fast list with max size (sliding window)
from typing import Any, Optional

# ── Try importing optional libraries ──────────────────────────────────────
# These are "optional" — the script works without them
# but some features won't be available

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[WARN] numpy not installed. Some features disabled.")
    print("       Install with: pip3 install numpy")

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False
    print("[WARN] influxdb-client not installed. Dashboard disabled.")
    print("       Install with: pip3 install influxdb-client")

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("[INFO] TFLite not installed. Running in observation mode.")
        print("       Install with: pip3 install tflite-runtime")


# ============================================================
#  CONFIGURATION — Change these settings to match your setup
# ============================================================

# --- Serial Port ---
# This is where the ESP32 is connected.
# On Linux/Mac: usually "/dev/ttyUSB0" or "/dev/ttyACM0"
# On Windows:   usually "COM3" or "COM4"
# Run "ls /dev/tty*" in terminal to find yours.
SERIAL_PORT     = "/dev/ttyUSB0"
SERIAL_BAUD     = 115200          # Must match firmware (115200)
SERIAL_TIMEOUT  = 3               # Seconds to wait for data

# --- Data Storage ---
DATA_DIR        = "./aetherflow_data"          # Folder to save all data
CSV_FILENAME    = "sensor_log.csv"             # Main data log file
SESSION_LOG     = "session_log.txt"            # Human-readable log

# --- AI Model ---
MODEL_PATH      = "./model/aetherflow_model.tflite"  # Trained model file
AI_MODE         = False           # Set to True after you have a trained model
WINDOW_SIZE     = 120             # How many readings to feed the AI (120 × 0.5s = 60 seconds)

# --- Control Limits (safety bounds for AI commands) ---
PELTIER_MIN     = 0               # Minimum Peltier duty (0 = off)
PELTIER_MAX     = 220             # Maximum Peltier duty (not full blast — leaves headroom)
FAN_MIN         = 40              # Fans never go below this (stall prevention)
FAN_MAX         = 255             # Maximum fan speed

# --- Target Temperature ---
TEMP_TARGET     = 45.0            # °C — what we're trying to maintain

# --- InfluxDB Dashboard (optional) ---
INFLUX_URL      = "http://localhost:8086"
INFLUX_TOKEN    = "your-token-here"       # Set this after installing InfluxDB
INFLUX_ORG      = "aetherflow"
INFLUX_BUCKET   = "cooling_data"

# --- Logging ---
LOG_INTERVAL    = 10              # Print summary to terminal every N readings
SAVE_INTERVAL   = 1               # Save to CSV every N readings (1 = every reading)


# ============================================================
#  GLOBAL STATE
# ============================================================

# A "sliding window" of the last 120 readings for the AI model
# deque with maxlen automatically drops old readings when full
sensor_window = deque(maxlen=WINDOW_SIZE)

# Statistics tracker (running averages)
stats = {
    "total_readings":   0,
    "session_start":    datetime.now(),
    "avg_temp":         0.0,
    "min_temp":         999.0,
    "max_temp":         -999.0,
    "total_kwh":        0.0,
    "ai_commands_sent": 0,
    "pid_cycles":       0,
    "safety_events":    0,
}

# Shared data between threads
latest_reading  = {}       # Most recent sensor reading
running         = True     # Set to False to stop all threads
serial_conn     = None     # Serial connection object
ai_model        = None     # Loaded TFLite model


# ============================================================
#  SETUP — Create folders and files
# ============================================================

def setup_directories() -> None:
    """Create data directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "model"), exist_ok=True)
    print(f"[SETUP] Data directory: {os.path.abspath(DATA_DIR)}")


def setup_csv() -> str:
    """
    Create the CSV file with headers if it doesn't exist.

    CSV = Comma-Separated Values — a simple spreadsheet format.
    Every row is one sensor reading. Every column is one value.
    Excel and Python pandas can both read this easily.

    Returns:
        str: Absolute path to the CSV file.
    """
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)

    if not os.path.exists(csv_path):
        # File doesn't exist — create it with column headers
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",        # When the reading was taken
                "unix_time",        # Seconds since 1970 (useful for ML)
                "temp_c",           # Target temperature °C
                "temp_ambient_c",   # Room temperature °C
                "humidity_pct",     # Relative humidity %
                "dew_point_c",      # Dew point °C
                "power_watts",      # Power draw W
                "voltage_v",        # Bus voltage V
                "peltier_duty",     # Peltier PWM 0-255
                "fan1_duty",        # Fan 1 PWM 0-255
                "fan2_duty",        # Fan 2 PWM 0-255
                "fan1_rpm",         # Fan 1 actual RPM
                "fan2_rpm",         # Fan 2 actual RPM
                "control_mode",     # "AI" or "PID"
                "system_safe",      # 1 = healthy, 0 = emergency
                "temp_error",       # temp - target (positive = too hot)
            ])
        print(f"[SETUP] Created new CSV: {csv_path}")
    else:
        # File exists — count existing rows
        with open(csv_path, "r") as f:
            rows = sum(1 for _ in f) - 1  # Subtract header row
        print(f"[SETUP] Appending to existing CSV ({rows} rows already saved)")

    return csv_path


def setup_serial() -> bool:
    """
    Open the serial connection to the ESP32.

    Serial communication is how the Pi and ESP32 talk over USB.
    Think of it like a very simple text chat — the ESP32 sends
    a line of text (JSON), the Pi reads it, and vice versa.

    Returns:
        bool: True if connection succeeded, False otherwise.
    """
    global serial_conn

    print(f"\n[SERIAL] Connecting to ESP32 on {SERIAL_PORT}...")
    print(f"[SERIAL] Baud rate: {SERIAL_BAUD}")

    try:
        serial_conn = serial.Serial(
            port      = SERIAL_PORT,
            baudrate  = SERIAL_BAUD,
            timeout   = SERIAL_TIMEOUT,
            bytesize  = serial.EIGHTBITS,
            parity    = serial.PARITY_NONE,
            stopbits  = serial.STOPBITS_ONE,
        )
        time.sleep(2)          # Wait for ESP32 to reset after connection
        serial_conn.flushInput()  # Clear any stale data in the buffer
        print(f"[SERIAL] ✅ Connected to ESP32!")
        return True

    except serial.SerialException as e:
        print(f"\n[ERROR] Could not open serial port: {e}")
        print(f"\n  Possible fixes:")
        print(f"  1. Check ESP32 is plugged in via USB")
        print(f"  2. Try:  ls /dev/tty*   to find the right port")
        print(f"  3. Edit SERIAL_PORT in this script to match")
        print(f"  4. On Linux, run:  sudo usermod -a -G dialout $USER")
        print(f"     (then log out and back in)")
        return False


def setup_influx() -> Any:
    """Connect to InfluxDB for the live Grafana dashboard.

    Returns:
        InfluxDB WriteApi instance, or None if unavailable.
    """
    if not INFLUX_AVAILABLE:
        return None

    try:
        client = InfluxDBClient(
            url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("[INFLUX] ✅ Connected to InfluxDB dashboard")
        return write_api
    except Exception as e:
        print(f"[INFLUX] Not available: {e}")
        return None


def load_ai_model() -> bool:
    """
    Load the trained TFLite AI model.

    TFLite = TensorFlow Lite — a compact format for running
    neural networks on small devices like Raspberry Pi.

    The model was trained on your CSV data and learned to
    predict the optimal cooling settings.

    Returns:
        bool: True if model loaded successfully, False otherwise.
    """
    global ai_model

    if not TFLITE_AVAILABLE:
        print("[AI] TFLite not available — running in observation mode")
        return False

    if not os.path.exists(MODEL_PATH):
        print(f"[AI] Model file not found at {MODEL_PATH}")
        print(f"[AI] Running in observation mode — collecting training data")
        return False

    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        ai_model = interpreter
        print(f"[AI] ✅ Model loaded: {MODEL_PATH}")

        # Print model input/output shapes (useful for debugging)
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"[AI] Input shape:  {input_details[0]['shape']}")
        print(f"[AI] Output shape: {output_details[0]['shape']}")
        return True

    except Exception as e:
        print(f"[AI] Failed to load model: {e}")
        return False


# ============================================================
#  READING DATA FROM ESP32
# ============================================================

def read_serial_line() -> Optional[dict]:
    """
    Read one line of JSON from the ESP32.

    The ESP32 sends lines like this every 500ms:
    {"t":47.3,"ta":24.1,"h":52.1,"dp":14.2,"w":12.4,...}

    We read the line, parse the JSON, and return a dict.

    Returns:
        dict with sensor values, or None if reading fails.
    """
    if not serial_conn or not serial_conn.is_open:
        return None

    try:
        # Read until newline character
        raw_line = serial_conn.readline()

        # Decode bytes to string
        # 'utf-8' is the text encoding
        # 'ignore' skips any weird characters
        line = raw_line.decode("utf-8", errors="ignore").strip()

        # Skip empty lines and debug messages (don't start with {)
        if not line or not line.startswith("{"):
            return None

        # Parse JSON string into Python dictionary
        # e.g. '{"t":47.3}' → {"t": 47.3}
        data = json.loads(line)
        return data

    except json.JSONDecodeError:
        # Line wasn't valid JSON — happens occasionally, just skip it
        return None
    except Exception as e:
        print(f"[SERIAL] Read error: {e}")
        return None


def enrich_reading(data: dict) -> dict:
    """
    Add extra fields to the sensor reading before saving.

    The ESP32 sends the raw values. Here we add:
    - timestamp (human-readable date/time)
    - unix_time (seconds since epoch — useful for ML)
    - temp_error (how far from target we are)

    Args:
        data: Raw sensor dict from ESP32 with keys 't', 'ta', 'h', etc.

    Returns:
        Enriched dict with added 'timestamp', 'unix_time', 'temp_error' fields.
    """
    now = datetime.now()
    data["timestamp"]  = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    data["unix_time"]  = time.time()
    data["temp_error"] = round(data.get("t", 0) - TEMP_TARGET, 2)
    return data


# ============================================================
#  SAVING DATA
# ============================================================

def save_to_csv(data: dict, csv_path: str) -> None:
    """
    Append one reading to the CSV file.

    We use 'a' mode (append) so we never overwrite old data.
    Each call adds one row to the bottom of the spreadsheet.
    """
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.get("timestamp",    ""),
                data.get("unix_time",    ""),
                data.get("t",            ""),   # temp
                data.get("ta",           ""),   # ambient temp
                data.get("h",            ""),   # humidity
                data.get("dp",           ""),   # dew point
                data.get("w",            ""),   # power watts
                data.get("v",            ""),   # voltage
                data.get("p",            ""),   # peltier duty
                data.get("f1",           ""),   # fan1 duty
                data.get("f2",           ""),   # fan2 duty
                data.get("r1",           ""),   # fan1 rpm
                data.get("r2",           ""),   # fan2 rpm
                data.get("mode",         ""),   # control mode
                data.get("safe",         ""),   # system safe
                data.get("temp_error",   ""),   # temp error
            ])
    except Exception as e:
        print(f"[CSV] Write error: {e}")


def save_to_influx(data, write_api):
    """
    Send a data point to InfluxDB for the live Grafana dashboard.

    InfluxDB is a time-series database — it stores data points
    with timestamps and lets Grafana display them as live charts.
    """
    if not write_api:
        return

    try:
        point = (
            Point("aetherflow")
            .field("temperature",    data.get("t",  0))
            .field("temp_ambient",   data.get("ta", 0))
            .field("humidity",       data.get("h",  0))
            .field("dew_point",      data.get("dp", 0))
            .field("power_watts",    data.get("w",  0))
            .field("peltier_duty",   data.get("p",  0))
            .field("fan1_duty",      data.get("f1", 0))
            .field("fan2_duty",      data.get("f2", 0))
            .field("fan1_rpm",       data.get("r1", 0))
            .field("fan2_rpm",       data.get("r2", 0))
            .field("temp_error",     data.get("temp_error", 0))
            .tag("mode",             data.get("mode", "PID"))
            .tag("safe",             str(data.get("safe", 1)))
        )
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
    except Exception as e:
        pass   # Dashboard errors are non-critical — don't crash


# ============================================================
#  UPDATE STATISTICS
# ============================================================

def update_stats(data: dict) -> None:
    """Keep running statistics on temperature and energy."""
    temp = data.get("t", 0)
    power = data.get("w", 0)

    stats["total_readings"] += 1
    stats["avg_temp"] = (
        (stats["avg_temp"] * (stats["total_readings"] - 1) + temp)
        / stats["total_readings"]
    )
    stats["min_temp"] = min(stats["min_temp"], temp)
    stats["max_temp"] = max(stats["max_temp"], temp)

    # Accumulate energy: Power (W) × time (0.5s) ÷ 3600 = Wh
    stats["total_kwh"] += (power * 0.5) / 3_600_000

    if data.get("mode") == "AI":
        stats["ai_commands_sent"] += 1
    else:
        stats["pid_cycles"] += 1

    if not data.get("safe", 1):
        stats["safety_events"] += 1


# ============================================================
#  AI INFERENCE — Run model and send commands to ESP32
# ============================================================

def run_ai_inference() -> Optional[dict]:
    """
    Run the LSTM model on the last 60 seconds of sensor data
    and return optimal control commands.

    This is only called when:
    1. AI_MODE = True
    2. A trained model file exists
    3. We have at least WINDOW_SIZE readings collected

    Returns:
        dict: {"peltier": int, "fan1": int, "fan2": int, "mode": "AI"}
        or None if inference fails.
    """
    if not ai_model or len(sensor_window) < WINDOW_SIZE:
        return None

    if not NUMPY_AVAILABLE:
        return None

    try:
        # Convert sliding window to numpy array
        # Shape: (WINDOW_SIZE, num_features)
        features = [
            "t", "ta", "h", "dp", "w", "v",
            "p", "f1", "f2", "temp_error"
        ]

        window_data = []
        for reading in sensor_window:
            row = [reading.get(f, 0.0) for f in features]
            window_data.append(row)

        # Normalize input data to 0-1 range
        # (Neural networks work best with small numbers)
        X = np.array(window_data, dtype=np.float32)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        X_norm = (X - X_min) / X_range

        # Add batch dimension: (1, WINDOW_SIZE, num_features)
        X_input = X_norm[np.newaxis, :, :]

        # Run inference
        input_details  = ai_model.get_input_details()
        output_details = ai_model.get_output_details()

        ai_model.set_tensor(input_details[0]["index"], X_input)
        ai_model.invoke()

        output = ai_model.get_tensor(output_details[0]["index"])[0]

        # Output: [peltier_norm, fan_norm] (normalized 0-1)
        # Convert back to actual PWM values
        peltier = int(np.clip(output[0] * 255, PELTIER_MIN, PELTIER_MAX))
        fan_speed = int(np.clip(output[1] * 255, FAN_MIN, FAN_MAX))

        return {
            "peltier": peltier,
            "fan1":    fan_speed,
            "fan2":    fan_speed,
            "mode":    "AI"
        }

    except Exception as e:
        print(f"[AI] Inference error: {e}")
        return None


def send_ai_command(command: dict) -> None:
    """
    Send a JSON control command to the ESP32.

    The ESP32 reads this and immediately applies the
    Peltier and fan duty cycles.

    Args:
        command: Dict with keys 'peltier', 'fan1', 'fan2' (0–255 PWM values).
    """
    if not serial_conn or not serial_conn.is_open:
        return

    try:
        cmd_str = json.dumps(command) + "\n"
        serial_conn.write(cmd_str.encode("utf-8"))
    except Exception as e:
        print(f"[SERIAL] Send error: {e}")


# ============================================================
#  DISPLAY — Print live readings to terminal
# ============================================================

def print_live_reading(data: dict, reading_num: int) -> None:
    """Print a formatted live reading to the terminal."""

    # Only print every LOG_INTERVAL readings to avoid spam
    if reading_num % LOG_INTERVAL != 0:
        return

    temp     = data.get("t",    0)
    ambient  = data.get("ta",   0)
    humidity = data.get("h",    0)
    power    = data.get("w",    0)
    peltier  = data.get("p",    0)
    fan1     = data.get("f1",   0)
    mode     = data.get("mode", "?")
    safe     = data.get("safe", 1)
    error    = data.get("temp_error", 0)

    # Colored status indicators
    temp_status  = "🔴 HOT" if temp > 60 else ("🟡 WARM" if temp > TEMP_TARGET + 2 else "🟢 OK")
    safe_status  = "✅ SAFE" if safe else "🚨 EMERGENCY"
    mode_status  = "🤖 AI" if mode == "AI" else "⚙️  PID"

    print(f"\n{'─'*55}")
    print(f"  Aether-Flow  |  {data.get('timestamp', '')}")
    print(f"{'─'*55}")
    print(f"  Temp:       {temp:6.1f}°C  {temp_status}  (error: {error:+.1f}°C)")
    print(f"  Ambient:    {ambient:6.1f}°C")
    print(f"  Humidity:   {humidity:6.1f}%")
    print(f"  Power:      {power:6.1f}W")
    print(f"  Peltier:    {peltier:6d}/255  ({peltier/255*100:.0f}%)")
    print(f"  Fan 1/2:    {fan1:3d}/{data.get('f2',0):3d}  RPM: {data.get('r1',0):.0f}/{data.get('r2',0):.0f}")
    print(f"  Control:    {mode_status}   System: {safe_status}")
    print(f"  Readings:   {stats['total_readings']:,}  |  Energy: {stats['total_kwh']*1000:.3f} Wh")
    print(f"  Avg temp:   {stats['avg_temp']:.1f}°C  |  Min: {stats['min_temp']:.1f}°C  Max: {stats['max_temp']:.1f}°C")


def print_startup_banner() -> None:
    """Print a nice startup banner to the terminal."""
    print("\n" + "═"*55)
    print("  ░█▀█░█▀▀░▀█▀░█░█░█▀▀░█▀▄░░░█▀▀░█░░░█▀█░█░█")
    print("  ░█▀█░█▀▀░░█░░█▀█░█▀▀░█▀▄░░░█▀▀░█░░░█░█░█▄█")
    print("  ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀░▀░░░▀░░░▀▀▀░▀▀▀░▀░▀")
    print("  AI-Driven Molecular Cooling — Raspberry Pi")
    print("═"*55)
    print(f"  Serial port:   {SERIAL_PORT}")
    print(f"  Data folder:   {os.path.abspath(DATA_DIR)}")
    print(f"  Target temp:   {TEMP_TARGET}°C")
    print(f"  AI mode:       {'ENABLED' if AI_MODE else 'OBSERVATION (collecting data)'}")
    print(f"  Window size:   {WINDOW_SIZE} readings ({WINDOW_SIZE*0.5:.0f} seconds)")
    print("═"*55 + "\n")


# ============================================================
#  MAIN LOOP
# ============================================================

def main():
    """
    The main program loop.

    This function:
    1. Sets everything up
    2. Reads data from ESP32 in a loop
    3. Saves data, updates stats, runs AI if enabled
    4. Handles errors gracefully
    """
    global running

    print_startup_banner()

    # --- Setup ---
    setup_directories()
    csv_path  = setup_csv()
    influx_api = setup_influx()

    if not setup_serial():
        print("\n[FATAL] Could not connect to ESP32. Exiting.")
        sys.exit(1)

    # Load AI model if available
    model_loaded = load_ai_model()
    ai_active = AI_MODE and model_loaded

    print(f"\n[START] Data collection started. Press Ctrl+C to stop.\n")

    reading_count = 0

    # --- Main loop ---
    while running:
        try:
            # STEP 1: Read one sensor reading from ESP32
            data = read_serial_line()

            if data is None:
                # No data received — ESP32 might be busy or disconnected
                continue

            # STEP 2: Add timestamp and computed fields
            data = enrich_reading(data)

            # STEP 3: Add to sliding window for AI
            sensor_window.append(data)

            # STEP 4: Update statistics
            update_stats(data)
            reading_count += 1

            # STEP 5: Save to CSV
            if reading_count % SAVE_INTERVAL == 0:
                save_to_csv(data, csv_path)

            # STEP 6: Save to InfluxDB (dashboard)
            save_to_influx(data, influx_api)

            # STEP 7: Run AI inference and send command (if enabled)
            if ai_active:
                command = run_ai_inference()
                if command:
                    send_ai_command(command)

            # STEP 8: Print live reading to terminal
            print_live_reading(data, reading_count)

            # STEP 9: Store as latest reading (for other tools to access)
            latest_reading.update(data)

        except KeyboardInterrupt:
            # User pressed Ctrl+C — clean shutdown
            print("\n\n[STOP] Ctrl+C received — shutting down gracefully...")
            running = False
            break

        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            time.sleep(1)   # Wait a second before retrying

    # --- Cleanup ---
    print_session_summary()

    if serial_conn and serial_conn.is_open:
        serial_conn.close()
        print("[SERIAL] Connection closed.")

    print("[DONE] Aether-Flow data logger stopped.")


# ============================================================
#  SESSION SUMMARY
# ============================================================

def print_session_summary() -> None:
    """Print a summary of the data collection session."""
    duration = datetime.now() - stats["session_start"]
    hours = duration.total_seconds() / 3600

    print("\n" + "═"*55)
    print("  SESSION SUMMARY")
    print("═"*55)
    print(f"  Duration:        {duration}")
    print(f"  Total readings:  {stats['total_readings']:,}")
    print(f"  Readings/hour:   {stats['total_readings']/max(hours,0.001):.0f}")
    print(f"  Avg temperature: {stats['avg_temp']:.2f}°C")
    print(f"  Min / Max temp:  {stats['min_temp']:.1f}°C / {stats['max_temp']:.1f}°C")
    print(f"  Total energy:    {stats['total_kwh']*1000:.3f} Wh")
    print(f"  AI cycles:       {stats['ai_commands_sent']:,}")
    print(f"  PID cycles:      {stats['pid_cycles']:,}")
    print(f"  Safety events:   {stats['safety_events']}")
    print(f"  Data saved to:   {os.path.join(DATA_DIR, CSV_FILENAME)}")
    print("═"*55)


# ============================================================
#  BONUS: Data Quality Report
#  Run this separately to check your collected data
# ============================================================

def check_data_quality(csv_path: Optional[str] = None) -> None:
    """
    Analyze your collected CSV data and report:
    - How many rows collected
    - Any missing values
    - Temperature distribution
    - Whether you have enough data to train the AI

    Args:
        csv_path: Path to CSV file. Defaults to the standard data location.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Install pandas first: pip3 install pandas")
        return

    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, CSV_FILENAME)

    if not os.path.exists(csv_path):
        print(f"No data file found at {csv_path}")
        print("Run the logger first to collect data.")
        return

    df = pd.read_csv(csv_path)

    print("\n" + "═"*55)
    print("  AETHER-FLOW DATA QUALITY REPORT")
    print("═"*55)
    print(f"\n  File:           {csv_path}")
    print(f"  Total rows:     {len(df):,}")
    print(f"  Columns:        {len(df.columns)}")

    # Time span
    if "timestamp" in df.columns:
        print(f"  First reading:  {df['timestamp'].iloc[0]}")
        print(f"  Last reading:   {df['timestamp'].iloc[-1]}")

    # Duration
    hours = len(df) * 0.5 / 3600
    print(f"  Duration:       {hours:.1f} hours ({hours/24:.1f} days)")

    # Temperature stats
    if "temp_c" in df.columns:
        print(f"\n  Temperature stats:")
        print(f"    Mean:    {df['temp_c'].mean():.2f}°C")
        print(f"    Std dev: {df['temp_c'].std():.2f}°C")
        print(f"    Min/Max: {df['temp_c'].min():.1f}°C / {df['temp_c'].max():.1f}°C")

    # Missing values
    missing = df.isnull().sum().sum()
    print(f"\n  Missing values: {missing}")
    if missing > 0:
        print("  (Some sensor reads failed — normal if < 1% of rows)")

    # AI readiness
    print(f"\n  AI Training Readiness:")
    target_rows = 100_000   # ~14 hours of data at 2/sec
    progress = min(100, len(df) / target_rows * 100)
    bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
    print(f"  [{bar}] {progress:.0f}%  ({len(df):,} / {target_rows:,} rows)")

    if len(df) < 10_000:
        print("  ⏳ Keep collecting — need more data before training")
    elif len(df) < 50_000:
        print("  🟡 Enough for basic training — more data = better model")
    else:
        print("  ✅ Ready to train the AI model!")
        print("  Next step: Run AetherFlow_Train.py")

    print("═"*55)


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            # Run data quality check instead of logging
            check_data_quality()
            sys.exit(0)
        elif sys.argv[1] == "--port":
            # Override serial port from command line
            # Usage: python3 AetherFlow_DataLogger.py --port /dev/ttyACM0
            if len(sys.argv) > 2:
                SERIAL_PORT = sys.argv[2]
                print(f"[CONFIG] Using port: {SERIAL_PORT}")

    main()
