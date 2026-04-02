# Aether-Flow

### AI-Driven Predictive Thermal Management via Power-Draw Precursor Signals

![License](https://img.shields.io/github/license/singhhrishabh/Aether-flow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)
![Status](https://img.shields.io/badge/status-Pre--hardware%20validation-yellow)

> Aether-Flow predicts thermal events **15–30 seconds before they occur**
> by reading INA226 power draw sensors and running a Temporal Fusion
> Transformer model entirely on-device on a Raspberry Pi 5 — then
> pre-emptively activates Peltier cooling modules before temperature rises.
> No internet. No cloud. Fully offline.

---

## Abstract

This project proposes a predictive thermal management system that exploits the temporal lag between electrical power draw and thermal response in computing systems. By deploying INA226 current sensors alongside a Temporal Fusion Transformer (TFT) model running inference on a Raspberry Pi 5 (<30 ms), the system predicts temperature excursions 15–30 seconds before onset and pre-activates Peltier thermoelectric coolers via ESP32 PWM control.

A hardware PID controller (Kp = 2.5, Ki = 0.1, Kd = 1.0) provides a safety fallback, while a PPO reinforcement-learning agent optimises multi-zone PWM allocation. Safety subsystems — including dew-point condensation prevention, thermal cutoff, and relay-based power isolation — ensure operation within safe bounds.

**Current status:** All software, firmware, and circuit designs are complete and simulation-validated. Hardware prototype construction is pending lab access.

---

## Research Questions

1. **RQ1 — Precursor Signal Validity:** Can electrical current draw serve as a reliable 15–30 second precursor signal for temperature rise in multi-core computing systems?
2. **RQ2 — Edge AI Feasibility:** Does a TFT model achieve sufficient prediction accuracy (<1 °C RMSE) within the latency budget (<30 ms) on edge hardware (Raspberry Pi 5)?
3. **RQ3 — Energy Efficiency:** Does predictive pre-cooling achieve measurable energy savings (>20 %) compared to reactive PID control under equivalent thermal loads?

---

## The Core Idea

Every cooling system today reacts **after** temperature rises. Aether-Flow acts **before** — by reading the power draw signal that precedes heat by 5–15 seconds and predicting the outcome.

```
Normal:      [heat rises] → [sensor detects] → [cooling starts]  ← too late
Aether-Flow: [power rises] → [AI predicts]   → [cooling starts]  ← before heat arrives
```

---

## Live Simulations

![Aether-Flow — AI predicts heat before it arrives](demo.gif)

*Watch the blue dashed line rise BEFORE actual temperature moves — that is the AI predicting the future and cooling starting early.*

Open directly in browser — no server needed:

- [Full AI Dashboard](https://singhhrishabh.github.io/Aether-flow/simulation/AetherFlow_WorkingModel.html)
- [How It Works Animation](https://singhhrishabh.github.io/Aether-flow/simulation/AetherFlow_HowItWorks.html)
- [3D Assembly Animation](https://singhhrishabh.github.io/Aether-flow/simulation/AetherFlow_Assembly.html)
- [6-Layer Architecture](https://singhhrishabh.github.io/Aether-flow/simulation/AetherFlow_LayerAnimation.html)
- [3D Component Catalogue](https://singhhrishabh.github.io/Aether-flow/simulation/Aetherflow_componentcatalogue.html)

| File | What it shows |
|---|---|
| `simulation/AetherFlow_WorkingModel.html` | Full live AI dashboard with thermal camera, prediction timeline, Peltier control |
| `simulation/AetherFlow_HowItWorks.html` | Animated explainer of the full system |
| `simulation/AetherFlow_Assembly.html` | 3D assembly animation |
| `simulation/AetherFlow_LayerAnimation.html` | 6-layer architecture walkthrough |
| `simulation/Aetherflow_componentcatalogue.html` | 3D component catalogue with specs |

---

## Related Work

| System | Approach | Limitation Aether-Flow Addresses |
|---|---|---|
| Lazic et al. (2018) — DeepMind / Google | RL-based data-centre cooling | Facility-level only; requires cloud connectivity; not edge-deployable |
| Intel DTPM (2021) | Dynamic thermal & power management via DVFS | Reactive, not predictive; no thermal forecasting horizon |
| Zhang et al. (2023) | LSTM-based server thermal prediction | Server-grade GPU required; not edge-deployable on RPi-class hardware |
| Traditional fan + heatsink | Passive / reactive PID | No prediction capability; responds only after thermal event occurs |
| Commercial BMS | Rule-based thermal management | No learning; no predictive horizon; limited to pre-set thresholds |

**Key differentiation:** Aether-Flow combines (1) a power-draw precursor signal with (2) on-device TFT inference on edge hardware, operating (3) fully offline with (4) device-level granularity — a combination not addressed by existing systems.

---

## Methodology

### Sensing Architecture

The system employs a multi-modal sensing approach:

- **Thermal sensing:** 6× TMP117 (±0.1 °C accuracy) for zone-level temperature mapping
- **Thermal imaging:** MLX90640 IR camera (32×24 px resolution) for surface heat distribution
- **Power monitoring:** 4× INA226 on CPU, GPU, memory, and system rail (500 ms polling via I²C)
- **Environmental:** SHT40 humidity sensor for dew-point calculation and condensation prevention
- **Flow sensing:** YF-S201 flow sensor for coolant loop monitoring

### Predictive Model

| Parameter | Value |
|---|---|
| Architecture | Temporal Fusion Transformer (TFT) |
| Input | 60-second sliding window (120 samples × 10 features) |
| Output | Temperature predictions at t+5 s, t+10 s, t+15 s, t+20 s, t+30 s |
| Deployment format | TFLite INT8 quantised |
| Target inference latency | <30 ms on Raspberry Pi 5 |
| Training data target | ≥100,000 rows of synchronised sensor telemetry |

### Control Strategy

- **Primary:** PPO reinforcement-learning agent for optimal multi-zone PWM allocation
- **Fallback:** Hardware PID controller (Kp = 2.5, Ki = 0.1, Kd = 1.0) with anti-windup (±100 integral clamp)
- **Safety:** Dew-point guard (+5 °C buffer), thermal cutoff at 80 °C, relay-based power isolation with <50 ms response

### Data Pipeline

```
ESP32 (500 ms JSON telemetry via USB serial)
  → Raspberry Pi (AetherFlow_DataLogger.py)
    → CSV file (training dataset)
    → InfluxDB → Grafana (live monitoring)
    → TFLite inference → control command → back to ESP32
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  PERCEPTION LAYER                   │
│  TMP117 ×6 · MLX90640 thermal cam · INA226 ×4       │
│  SHT40 humidity · YF-S201 flow sensor               │
│                   I2C bus — 500ms                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               INTELLIGENCE LAYER                    │
│         Raspberry Pi 5 — fully offline              │
│   TFT model (TFLite) → temperature prediction       │
│   PPO RL agent → optimal PWM per zone               │
│         USB serial → ESP32 commands                 │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 ACTUATION LAYER                     │
│   ESP32 → IRF540N MOSFET × 4 → TEC1-12706 × 4       │
│   Kamoer NKP pump · Noctua fans · Safety relay      │
│         Hardware PID fallback if AI offline         │
└─────────────────────────────────────────────────────┘
```

---

## Target Performance

| Metric | Target | Basis | Confidence |
|---|---|---|---|
| Temperature stability | ±0.3 °C | TMP117 sensor spec + PID simulation | Medium |
| AI inference latency | <30 ms | TFLite benchmark on RPi 5 (published) | High |
| Predictive horizon | 15–30 s | INA226 power-to-thermal lag measurement | Medium |
| Energy savings vs PID | 35–41 % | Simulation model comparison | Low — needs validation |
| Acoustic noise | ~28 dB | Noctua NF-A12x25 datasheet spec | High (component-level) |
| Safety response | <50 ms | ESP32 ISR + relay switching time | High |

> **Honest note:** All system-level performance figures are design targets from simulation.
> Physical hardware validation is the next step and is the primary research objective.

---

## Experimental Plan

### Phase 1 — Data Collection (Weeks 1–4)
- Deploy sensing array on a test rig (Raspberry Pi 5 under synthetic workloads: stress-ng, sysbench)
- Collect ≥100,000 synchronised sensor readings (temperature, power, humidity, fan RPM)
- Validate INA226-to-temperature lag hypothesis with cross-correlation analysis
- Characterise per-zone thermal response curves under step-load inputs

### Phase 2 — Model Training & Validation (Weeks 5–8)
- Train TFT model on collected data with 80/20 temporal split (no shuffling — time preserved)
- Evaluate RMSE, MAE, and R² across all five prediction horizons (+5 s to +30 s)
- Benchmark inference latency on Raspberry Pi 5 (TFLite INT8 quantised)
- Compare TFT against LSTM, GRU, and simple linear baselines

### Phase 3 — Closed-Loop Testing (Weeks 9–12)
- Deploy full system with Peltier + fan actuation
- A/B test: predictive AI control vs. reactive PID under identical workloads
- Measure energy consumption, thermal stability (σ_T), and response latency
- Validate safety subsystems (dew-point guard, thermal cutoff, relay response)

### Phase 4 — Analysis & Publication (Weeks 13–16)
- Statistical analysis with confidence intervals
- Energy savings quantification (kWh comparison)
- Write-up targeting IEEE/ACM embedded systems or thermal management venue

---

## Current Status

- [x] All 4 subsystem circuits simulated and verified in Tinkercad
- [x] ESP32 firmware written — ready for hardware deployment
- [x] Raspberry Pi 5 data logger written (Python 3.11)
- [x] TFLite model training pipeline complete (PyTorch + PyTorch Forecasting)
- [x] Interactive AI simulation with predictive vs reactive comparison
- [x] Full project proposal with BOM, architecture, market analysis
- [ ] Physical hardware prototype — seeking lab access and funding
- [ ] Real sensor data collection (target: 100,000+ rows)
- [ ] TFT model training on real hardware data
- [ ] Performance validation against design targets

---

## Repository Structure

```
aether-flow/
├── firmware/
│   ├── AetherFlow_Firmware.ino          ESP32 firmware (Arduino/FreeRTOS)
│   └── AetherFlow_FirmwareGuide.md      Wiring & flash guide
├── software/
│   ├── AetherFlow_DataLogger.py         Raspberry Pi sensor logger
│   ├── AetherFlow_Train.py              TFLite model training pipeline
│   └── requirements.txt                 Pinned Python dependencies
├── circuits/
│   ├── AetherFlow_Circuits.html         Interactive circuit viewer
│   ├── circuit1_temperature_sensing.png
│   ├── circuit2_mosfet_pwm.png
│   ├── circuit3_pid_controller.png
│   └── circuit4_safety_relay.png
├── simulation/
│   ├── AetherFlow_WorkingModel.html     Live AI cooling simulation
│   ├── AetherFlow_HowItWorks.html       Animated system explainer
│   ├── AetherFlow_Assembly.html         3D assembly animation
│   ├── AetherFlow_LayerAnimation.html   6-layer architecture walkthrough
│   └── Aetherflow_componentcatalogue.html  3D component catalogue
├── tests/
│   ├── test_data_logger.py              Unit tests for data pipeline
│   └── test_train_pipeline.py           Unit tests for training pipeline
├── data/
│   ├── README.md                        Data dictionary
│   └── sample_synthetic.csv             1000-row sample for testing
├── docs/
│   ├── ARCHITECTURE.md                  Detailed system architecture
│   └── EXPERIMENT_PROTOCOL.md           Step-by-step experiment methodology
├── .github/
│   └── workflows/
│       └── ci.yml                       Lint & test automation
├── CITATION.cff                         Citation metadata
├── pyproject.toml                       Python project metadata
└── README.md
```

---

## Hardware Bill of Materials

| Component | Qty | Role | Est. Cost (USD) |
|---|---|---|---|
| Raspberry Pi 5 4 GB | 1 | AI inference | $60 |
| ESP32-WROOM-32 | 1 | PWM control + PID fallback | $8 |
| TEC1-12706 Peltier module | 4 | Thermoelectric cooling | $20 |
| TMP117 temperature sensor | 6 | Zone-level temperature (±0.1 °C) | $30 |
| MLX90640 IR thermal camera | 1 | Surface heat distribution | $70 |
| INA226 power monitor | 4 | AI input — power precursor signal | $16 |
| SHT40 humidity sensor | 1 | Dew-point calculation | $8 |
| IRF540N N-channel MOSFET | 4 | Peltier PWM switching | $4 |
| Noctua NF-A12x25 fan | 2 | Active cooling | $60 |
| Kamoer NKP peristaltic pump | 1 | Coolant loop | $35 |
| 12 V 30 A power supply | 1 | System power | $25 |
| Safety relay module | 1 | Emergency power isolation | $5 |
| YF-S201 flow sensor | 1 | Coolant flow monitoring | $4 |
| Miscellaneous (PCB, wiring, connectors) | — | — | $14 |

**Total BOM: ~$359 USD**

---

## Competitive Position

| Capability | Fan + Heatsink | Google DeepMind AI | Commercial BMS | **Aether-Flow** |
|---|---|---|---|---|
| Predictive | No | Yes (facility level) | No | **Yes (device level)** |
| Fully offline | Yes | No | Yes | **Yes** |
| Self-learning | No | Yes | No | **Yes** |
| <30 ms inference | N/A | No | N/A | **Yes** |
| Thermal camera | No | No | No | **Yes** |
| Device-level | Yes | No | Partial | **Yes** |

---

## Markets Addressed

- **EV battery thermal management** — $58.9 B by 2035 (16.3 % CAGR)
- **Data-centre liquid cooling** — $38.4 B by 2033 (28.7 % CAGR)
- **AI-driven BMS** — $18.5 B by 2032 (20.6 % CAGR)
- **Edge AI / Industrial IoT** — $18 B by 2028
- **5G base stations** — $15 B addressable

---

## Getting Started

### Prerequisites

- Python 3.11+ on Raspberry Pi 5
- Arduino IDE 2.x for ESP32 firmware
- (Optional) InfluxDB + Grafana for live dashboard

### Installation

```bash
# Clone the repository
git clone https://github.com/singhhrishabh/Aether-flow.git
cd Aether-flow

# Install Python dependencies
pip3 install -r software/requirements.txt

# Flash firmware to ESP32
# Open firmware/AetherFlow_Firmware.ino in Arduino IDE
# Select Board: ESP32-WROOM-32 → Upload

# Start data collection
python3 software/AetherFlow_DataLogger.py --port /dev/ttyUSB0

# Check data quality
python3 software/AetherFlow_DataLogger.py --check

# Train the AI model (after collecting ≥50,000 rows)
python3 software/AetherFlow_Train.py
```

### Running Tests

```bash
pip3 install pytest numpy pandas
pytest tests/ -v
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{singh2026aetherflow,
  author    = {Singh, Rishabh},
  title     = {Aether-Flow: AI-Driven Predictive Thermal Management},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/singhhrishabh/Aether-flow}
}
```

---

## Author

**Rishabh Singh**  
B.E. ECE Year 1 — BITS Pilani Dubai Campus  
📧 [rishabh.s0072@gmail.com](mailto:rishabh.s0072@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/rishabh-singh-8b66003b2/)  

*Seeking research supervision and lab access to build and validate the physical prototype.*

[GitHub](https://github.com/singhhrishabh) | [Live Demo](https://singhhrishabh.github.io/Aether-flow/)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
