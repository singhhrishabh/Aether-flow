# Aether-Flow — Experiment Protocol

## Objective

Validate the hypothesis that electrical power draw can serve as a 15–30 second precursor signal for temperature rise, enabling predictive pre-cooling with measurable energy savings over reactive PID control.

## Equipment Required

| Item | Specification | Purpose |
|---|---|---|
| Raspberry Pi 5 (4 GB) | Running Raspberry Pi OS Bookworm, Python 3.11 | AI inference + data logging |
| ESP32-WROOM-32 | Arduino IDE 2.x, firmware v1.0 | Sensor polling + PWM actuation |
| TMP117 sensors ×6 | I²C addresses 0x48–0x4B | Zone temperature measurement |
| INA226 monitors ×4 | I²C addresses 0x40–0x43 | Power draw measurement (AI input) |
| SHT40 sensor | I²C address 0x44 | Humidity + dew-point calculation |
| TEC1-12706 modules ×4 | 12 V, ΔT_max 66 °C | Active cooling |
| 12 V / 30 A PSU | Regulated | System power |
| USB cable (USB-A to Micro-B) | Data-capable | ESP32 ↔ RPi serial link |

## Phase 1 — Data Collection

### Duration: 4 weeks

### Procedure

1. Assemble the hardware rig per `docs/ARCHITECTURE.md` wiring guide
2. Flash `firmware/AetherFlow_Firmware.ino` to ESP32
3. Connect ESP32 to Raspberry Pi 5 via USB
4. Start the data logger:
   ```bash
   python3 software/AetherFlow_DataLogger.py --port /dev/ttyUSB0
   ```
5. Generate synthetic workloads using `stress-ng`:
   ```bash
   # CPU stress (varies intensity)
   stress-ng --cpu 4 --cpu-load 25 --timeout 120s
   stress-ng --cpu 4 --cpu-load 50 --timeout 120s
   stress-ng --cpu 4 --cpu-load 75 --timeout 120s
   stress-ng --cpu 4 --cpu-load 100 --timeout 120s
   ```
6. Run idle → load → idle cycles (10 min each phase) to capture transients
7. Collect data 24/7 for at least 2 weeks (target: ≥100,000 rows)

### Validation Checks (Weekly)

```bash
python3 software/AetherFlow_DataLogger.py --check
```

- Verify row count progress
- Check for missing values (acceptable: <1 % of rows)
- Inspect temperature range (expected: 25–75 °C)
- Confirm power-to-temperature lag with cross-correlation

### Cross-Correlation Analysis

After collecting ≥10,000 rows:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("aetherflow_data/sensor_log.csv")

# Cross-correlate power draw with temperature
power = df["power_watts"].values
temp = df["temp_c"].values

lags = range(-60, 60)  # ±30 seconds at 2 Hz
correlations = [np.corrcoef(power[60:-60], temp[60+lag:-60+lag or None])[0,1]
                for lag in lags]

peak_lag = lags[np.argmax(correlations)]
print(f"Peak correlation at lag = {peak_lag} samples ({peak_lag * 0.5}s)")
# Expected: peak at lag 10–60 samples (5–30 seconds)
```

## Phase 2 — Model Training

### Duration: 4 weeks

### Procedure

1. Transfer CSV data to a training machine (GPU recommended)
2. Run the training pipeline:
   ```bash
   python3 software/AetherFlow_Train.py
   ```
3. Record metrics:
   - Training/validation loss curves
   - Final validation RMSE, MAE, R²
   - Model file size (KB)
4. Copy `.tflite` model back to Raspberry Pi:
   ```bash
   scp aetherflow_data/model/aetherflow_model.tflite pi@raspberrypi:~/Aether-flow/aetherflow_data/model/
   ```
5. Benchmark inference latency on RPi 5:
   ```python
   import time
   import tflite_runtime.interpreter as tflite
   import numpy as np

   interpreter = tflite.Interpreter(model_path="aetherflow_model.tflite")
   interpreter.allocate_tensors()

   input_details = interpreter.get_input_details()
   dummy_input = np.random.rand(1, 120, 10).astype(np.float32)

   times = []
   for _ in range(100):
       start = time.perf_counter()
       interpreter.set_tensor(input_details[0]["index"], dummy_input)
       interpreter.invoke()
       times.append((time.perf_counter() - start) * 1000)

   print(f"Inference: {np.mean(times):.1f} ± {np.std(times):.1f} ms")
   ```

### Baseline Comparisons

Train and evaluate alternative architectures for comparison:

| Model | Architecture | Expected Inference |
|---|---|---|
| TFT (primary) | Temporal Fusion Transformer | <30 ms |
| LSTM | 2-layer LSTM (64→32) | <20 ms |
| GRU | 2-layer GRU (64→32) | <15 ms |
| Linear | Ridge regression on flattened window | <5 ms |

## Phase 3 — Closed-Loop Testing

### Duration: 4 weeks

### Procedure

1. Enable AI mode:
   ```python
   # In AetherFlow_DataLogger.py, set:
   AI_MODE = True
   ```
2. Run identical workload sequences under:
   - **Condition A:** PID-only control (AI_MODE = False)
   - **Condition B:** AI predictive control (AI_MODE = True)
3. Each condition: 5 repetitions × 1-hour sessions

### Metrics to Record

| Metric | Measurement | Tool |
|---|---|---|
| Energy consumption | Total kWh over session | INA226 cumulative |
| Thermal stability | σ_T (standard deviation of temperature) | CSV analysis |
| Response latency | Time from load start to first cooling action | CSV timestamp delta |
| Overshoot | Max temperature excursion above target | CSV max(temp_c) |
| Safety events | Count of dew-point guard / thermal cutoff triggers | Logger stats |

### Statistical Analysis

```python
from scipy import stats

# Compare energy consumption: PID vs AI
pid_energy = [...]  # kWh from 5 PID sessions
ai_energy = [...]   # kWh from 5 AI sessions

t_stat, p_value = stats.ttest_ind(pid_energy, ai_energy)
effect_size = (np.mean(pid_energy) - np.mean(ai_energy)) / np.std(pid_energy)

print(f"Energy savings: {(1 - np.mean(ai_energy)/np.mean(pid_energy)) * 100:.1f}%")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {effect_size:.2f}")
```

## Phase 4 — Analysis & Write-Up

### Duration: 4 weeks

### Deliverables

1. **Figures:** Training curves, comparison bar charts, thermal response traces
2. **Tables:** Per-horizon prediction accuracy, energy comparison, latency benchmark
3. **Statistical tests:** t-tests with effect sizes for all primary metrics
4. **Discussion:** Limitations, failure modes, scalability analysis
5. **Paper target:** IEEE Embedded Systems Letters or ACM TECS
