# Aether-Flow — Sample Data

## Data Dictionary

This directory contains sample synthetic data for testing the training pipeline
without requiring physical hardware.

### CSV Column Definitions

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | string | ISO 8601 | Human-readable datetime of reading |
| `unix_time` | float | seconds | Unix epoch timestamp (for ML) |
| `temp_c` | float | °C | Target zone temperature |
| `temp_ambient_c` | float | °C | Room/ambient temperature |
| `humidity_pct` | float | % | Relative humidity |
| `dew_point_c` | float | °C | Calculated dew point |
| `power_watts` | float | W | System power draw (AI precursor signal) |
| `voltage_v` | float | V | Bus voltage |
| `peltier_duty` | int | 0–255 | Peltier PWM duty cycle |
| `fan1_duty` | int | 0–255 | Fan 1 PWM duty cycle |
| `fan2_duty` | int | 0–255 | Fan 2 PWM duty cycle |
| `fan1_rpm` | float | RPM | Fan 1 measured speed |
| `fan2_rpm` | float | RPM | Fan 2 measured speed |
| `control_mode` | string | — | `"AI"` or `"PID"` |
| `system_safe` | int | 0/1 | 1 = healthy, 0 = emergency stop |
| `temp_error` | float | °C | `temp_c - 45.0` (target = 45 °C) |

### `sample_synthetic.csv`

- **1,000 rows** of synthetically generated data
- Simulates an idle → load → idle → load cycle
- Power draw leads temperature rise by ~20 samples (10 seconds)
- Peltier duty follows a simple proportional response
- **Not real data** — for pipeline testing only

### Generating Your Own Sample Data

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
t = np.arange(n) * 0.5  # 500ms intervals

# Simulate workload cycles
load = np.sin(2 * np.pi * t / 300) * 0.5 + 0.5  # ~5-min cycles
power = 5 + load * 20 + np.random.normal(0, 0.5, n)
temp = 30 + np.convolve(load, np.ones(40)/40, mode='same') * 25 + np.random.normal(0, 0.3, n)

df = pd.DataFrame({
    "timestamp": pd.date_range("2026-01-01", periods=n, freq="500ms"),
    "unix_time": t + 1735689600,
    "temp_c": temp.round(2),
    "temp_ambient_c": (25 + np.random.normal(0, 0.2, n)).round(2),
    "humidity_pct": (50 + np.random.normal(0, 2, n)).round(1),
    "dew_point_c": (14 + np.random.normal(0, 0.5, n)).round(2),
    "power_watts": power.round(2),
    "voltage_v": (12.1 + np.random.normal(0, 0.05, n)).round(2),
    "peltier_duty": np.clip((temp - 45) * 10, 0, 255).astype(int),
    "fan1_duty": np.clip((temp - 40) * 8, 40, 255).astype(int),
    "fan2_duty": np.clip((temp - 40) * 8, 40, 255).astype(int),
    "fan1_rpm": np.clip((temp - 40) * 50, 300, 2400).round(0),
    "fan2_rpm": np.clip((temp - 40) * 50, 280, 2380).round(0),
    "control_mode": "PID",
    "system_safe": 1,
    "temp_error": (temp - 45).round(2),
})
df.to_csv("sample_synthetic.csv", index=False)
```
