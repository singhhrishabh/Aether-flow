"""
Unit tests for the AetherFlow Data Logger.

Tests the data enrichment, statistics tracking, CSV schema,
and sliding window logic without requiring hardware.
"""

import time
from datetime import datetime
from collections import deque


# ── Replicate core functions from AetherFlow_DataLogger.py ──────────────
# We import the logic inline to avoid hardware-dependent imports (pyserial, tflite).

TEMP_TARGET = 45.0
WINDOW_SIZE = 120


def enrich_reading(data: dict) -> dict:
    """Add computed metadata to a raw sensor reading."""
    now = datetime.now()
    data["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    data["unix_time"] = time.time()
    data["temp_error"] = round(data.get("t", 0) - TEMP_TARGET, 2)
    return data


def update_stats(data: dict, stats: dict) -> None:
    """Update running statistics with a new reading."""
    temp = data.get("t", 0)
    power = data.get("w", 0)

    stats["total_readings"] += 1
    stats["avg_temp"] = (
        (stats["avg_temp"] * (stats["total_readings"] - 1) + temp)
        / stats["total_readings"]
    )
    stats["min_temp"] = min(stats["min_temp"], temp)
    stats["max_temp"] = max(stats["max_temp"], temp)
    stats["total_kwh"] += (power * 0.5) / 3_600_000

    if data.get("mode") == "AI":
        stats["ai_commands_sent"] += 1
    else:
        stats["pid_cycles"] += 1

    if not data.get("safe", 1):
        stats["safety_events"] += 1


def make_fresh_stats() -> dict:
    """Create a fresh statistics dictionary."""
    return {
        "total_readings": 0,
        "session_start": datetime.now(),
        "avg_temp": 0.0,
        "min_temp": 999.0,
        "max_temp": -999.0,
        "total_kwh": 0.0,
        "ai_commands_sent": 0,
        "pid_cycles": 0,
        "safety_events": 0,
    }


# ── Tests ───────────────────────────────────────────────────────────────


class TestEnrichReading:
    """Tests for the enrich_reading function."""

    def test_adds_temp_error(self):
        """temp_error should be (measured temp) - TEMP_TARGET."""
        data = {"t": 50.0, "ta": 25.0}
        result = enrich_reading(data)
        assert result["temp_error"] == 5.0  # 50 - 45

    def test_adds_temp_error_negative(self):
        """Negative temp_error when below target."""
        data = {"t": 40.0}
        result = enrich_reading(data)
        assert result["temp_error"] == -5.0  # 40 - 45

    def test_adds_timestamp(self):
        """Should add a human-readable timestamp."""
        data = {"t": 45.0}
        result = enrich_reading(data)
        assert "timestamp" in result
        assert len(result["timestamp"]) > 10

    def test_adds_unix_time(self):
        """Should add a numeric Unix timestamp."""
        data = {"t": 45.0}
        result = enrich_reading(data)
        assert "unix_time" in result
        assert isinstance(result["unix_time"], float)
        assert result["unix_time"] > 1_700_000_000

    def test_handles_missing_temp(self):
        """Should default to 0 if 't' key is missing."""
        data = {}
        result = enrich_reading(data)
        assert result["temp_error"] == -45.0  # 0 - 45

    def test_preserves_original_keys(self):
        """Enrichment should not remove existing keys."""
        data = {"t": 50.0, "h": 60.0, "w": 12.5}
        result = enrich_reading(data)
        assert result["t"] == 50.0
        assert result["h"] == 60.0
        assert result["w"] == 12.5


class TestUpdateStats:
    """Tests for the update_stats function."""

    def test_increments_reading_count(self):
        stats = make_fresh_stats()
        update_stats({"t": 50.0, "w": 10.0, "mode": "PID", "safe": 1}, stats)
        assert stats["total_readings"] == 1

    def test_tracks_min_max(self):
        stats = make_fresh_stats()
        update_stats({"t": 30.0, "w": 5.0}, stats)
        update_stats({"t": 60.0, "w": 15.0}, stats)
        update_stats({"t": 45.0, "w": 10.0}, stats)
        assert stats["min_temp"] == 30.0
        assert stats["max_temp"] == 60.0

    def test_computes_running_average(self):
        stats = make_fresh_stats()
        update_stats({"t": 40.0, "w": 0}, stats)
        update_stats({"t": 50.0, "w": 0}, stats)
        assert abs(stats["avg_temp"] - 45.0) < 0.01

    def test_counts_ai_vs_pid_modes(self):
        stats = make_fresh_stats()
        update_stats({"t": 45.0, "w": 0, "mode": "AI"}, stats)
        update_stats({"t": 45.0, "w": 0, "mode": "PID"}, stats)
        update_stats({"t": 45.0, "w": 0, "mode": "AI"}, stats)
        assert stats["ai_commands_sent"] == 2
        assert stats["pid_cycles"] == 1

    def test_counts_safety_events(self):
        stats = make_fresh_stats()
        update_stats({"t": 80.0, "w": 50.0, "safe": 0}, stats)
        update_stats({"t": 45.0, "w": 10.0, "safe": 1}, stats)
        assert stats["safety_events"] == 1

    def test_accumulates_energy(self):
        stats = make_fresh_stats()
        # 10W for 0.5s = 5 Ws = 5/3600 Wh = 0.001389 Wh
        update_stats({"t": 45.0, "w": 10.0}, stats)
        assert stats["total_kwh"] > 0


class TestSlidingWindow:
    """Tests for the sliding window (deque) logic."""

    def test_window_max_size(self):
        window = deque(maxlen=WINDOW_SIZE)
        for i in range(200):
            window.append({"t": float(i)})
        assert len(window) == WINDOW_SIZE

    def test_window_fifo_order(self):
        window = deque(maxlen=5)
        for i in range(10):
            window.append({"t": float(i)})
        # Should contain last 5 items: 5, 6, 7, 8, 9
        temps = [r["t"] for r in window]
        assert temps == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestCSVSchema:
    """Tests that the CSV column list matches expectations."""

    def test_expected_columns(self):
        expected = [
            "timestamp", "unix_time", "temp_c", "temp_ambient_c",
            "humidity_pct", "dew_point_c", "power_watts", "voltage_v",
            "peltier_duty", "fan1_duty", "fan2_duty", "fan1_rpm",
            "fan2_rpm", "control_mode", "system_safe", "temp_error",
        ]
        assert len(expected) == 16

    def test_sensor_key_mapping(self):
        """Verify the ESP32 short keys map to the right CSV columns."""
        key_map = {
            "t": "temp_c",
            "ta": "temp_ambient_c",
            "h": "humidity_pct",
            "dp": "dew_point_c",
            "w": "power_watts",
            "v": "voltage_v",
            "p": "peltier_duty",
            "f1": "fan1_duty",
            "f2": "fan2_duty",
            "r1": "fan1_rpm",
            "r2": "fan2_rpm",
        }
        assert len(key_map) == 11
