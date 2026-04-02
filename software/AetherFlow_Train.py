#!/usr/bin/env python3
"""
============================================================
  AETHER-FLOW  —  AI Model Trainer
  File: AetherFlow_Train.py
============================================================

WHAT THIS SCRIPT DOES:
-----------------------
Takes your collected CSV data and trains a neural network
to predict the optimal cooling settings.

After training:
  - Saves the model as aetherflow_model.tflite
  - This file gets loaded by AetherFlow_DataLogger.py
  - The ESP32 then receives AI commands instead of using PID

WHEN TO RUN THIS:
-----------------
  1. After collecting at least 50,000 rows of sensor data
     (check with: python3 AetherFlow_DataLogger.py --check)

  2. Run on a PC/laptop for faster training (not on Pi)
     Then copy the .tflite file to the Pi.

  3. Re-run every few weeks as you collect more data

HOW TO RUN:
-----------
  pip3 install pandas numpy scikit-learn tensorflow
  python3 AetherFlow_Train.py

============================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ── Check dependencies ─────────────────────────────────────
missing = []
try:    import pandas as pd
except: missing.append("pandas")
try:    import numpy as np
except: missing.append("numpy")
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    missing.append("scikit-learn")
try:
    import tensorflow as tf
except ImportError:
    missing.append("tensorflow")

if missing:
    print(f"[ERROR] Missing libraries: {', '.join(missing)}")
    print(f"Install with: pip3 install {' '.join(missing)}")
    sys.exit(1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
#  CONFIG
# ============================================================

DATA_PATH    = "./aetherflow_data/sensor_log.csv"
MODEL_DIR    = "./aetherflow_data/model"
MODEL_NAME   = "aetherflow_model"

WINDOW_SIZE  = 120     # 60 seconds of data (120 × 0.5s)
BATCH_SIZE   = 64
EPOCHS       = 50
LEARNING_RATE= 0.001
TEMP_TARGET  = 45.0


# ============================================================
#  STEP 1 — LOAD AND PREPARE DATA
# ============================================================

def load_data():
    print("\n[1/5] Loading data...")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("Run AetherFlow_DataLogger.py first to collect data.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded {len(df):,} rows from {DATA_PATH}")

    if len(df) < 10_000:
        print(f"[WARN] Only {len(df):,} rows — model may be weak.")
        print("       Recommend 50,000+ rows for best results.")

    # Drop rows with missing values
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"      Dropped {dropped} rows with missing values")

    # Sort by time
    if "unix_time" in df.columns:
        df.sort_values("unix_time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
#  STEP 2 — FEATURE ENGINEERING
# ============================================================

def prepare_features(df):
    """
    Select which sensor columns to use as model inputs,
    and which to predict as outputs.

    INPUTS (what the model sees):
      - Temperature + history
      - Power draw (predicts future heat)
      - Humidity / dew point
      - Current fan/peltier state

    OUTPUTS (what the model predicts):
      - Optimal peltier duty
      - Optimal fan speed
    """
    print("\n[2/5] Preparing features...")

    # Input features
    input_cols = [
        "temp_c",           # Current temperature
        "temp_ambient_c",   # Room temperature
        "humidity_pct",     # Humidity
        "dew_point_c",      # Dew point
        "power_watts",      # Power draw
        "voltage_v",        # Voltage
        "peltier_duty",     # Current Peltier setting
        "fan1_duty",        # Current fan setting
        "temp_error",       # How far from target
    ]

    # Output targets (what AI should output)
    output_cols = ["peltier_duty", "fan1_duty"]

    # Check all columns exist
    for col in input_cols + output_cols:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' missing — filling with 0")
            df[col] = 0

    X_raw = df[input_cols].values.astype(np.float32)
    y_raw = df[output_cols].values.astype(np.float32)

    print(f"      Input shape:  {X_raw.shape}")
    print(f"      Output shape: {y_raw.shape}")
    print(f"      Features: {input_cols}")

    return X_raw, y_raw, input_cols, output_cols


# ============================================================
#  STEP 3 — CREATE SLIDING WINDOWS
# ============================================================

def create_windows(X, y, window_size):
    """
    Convert flat data into sliding windows for the LSTM.

    Instead of predicting from a single reading, the LSTM
    looks at the last 60 seconds of history.

    Before: each row = one moment in time
    After:  each sample = a window of WINDOW_SIZE moments
    """
    print(f"\n[3/5] Creating {window_size}-step sliding windows...")

    X_windows = []
    y_targets  = []

    for i in range(window_size, len(X)):
        # Input: window of past readings
        X_windows.append(X[i - window_size : i])
        # Target: the settings at the NEXT step
        # (what we should have done to maintain temperature)
        y_targets.append(y[i])

    X_windows = np.array(X_windows, dtype=np.float32)
    y_targets  = np.array(y_targets,  dtype=np.float32)

    print(f"      Window array shape: {X_windows.shape}")
    print(f"      Target array shape: {y_targets.shape}")

    return X_windows, y_targets


# ============================================================
#  STEP 4 — BUILD THE MODEL
# ============================================================

def build_model(window_size, n_features, n_outputs):
    """
    Build an LSTM neural network.

    Architecture:
      Input → LSTM → LSTM → Dense → Dense → Output

    LSTM = Long Short-Term Memory
    This type of layer remembers patterns over time sequences.
    Perfect for thermal prediction because past temperature
    history predicts future temperature trends.
    """
    print("\n[4/5] Building LSTM model...")

    model = tf.keras.Sequential([

        # Input layer — defines the shape of one sample
        tf.keras.layers.Input(shape=(window_size, n_features)),

        # First LSTM layer — learns temporal patterns
        # return_sequences=True means pass full sequence to next layer
        tf.keras.layers.LSTM(
            units=64,
            return_sequences=True,
            dropout=0.1,           # Dropout = randomly disable 10% of neurons during training
            recurrent_dropout=0.1  # Prevents overfitting (memorizing vs. learning)
        ),

        # Second LSTM layer — learns higher-level patterns
        tf.keras.layers.LSTM(
            units=32,
            return_sequences=False,  # Only output the final timestep
            dropout=0.1,
        ),

        # Dense layer — combines LSTM output into predictions
        tf.keras.layers.Dense(32, activation="relu"),  # relu = "only positive values"
        tf.keras.layers.Dropout(0.1),

        # Output layer — predicts [peltier_duty, fan_speed]
        # sigmoid activation = output between 0 and 1 (we scale to 0-255 later)
        tf.keras.layers.Dense(n_outputs, activation="sigmoid"),
    ])

    # Compile the model
    # optimizer = how the model updates its weights during training
    # loss = how we measure prediction error (MSE = mean squared error)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]  # Also track mean absolute error
    )

    model.summary()
    return model


# ============================================================
#  STEP 5 — TRAIN AND SAVE
# ============================================================

def train_and_save(model, X_train, y_train, X_val, y_val):
    """Train the model and save it as TFLite."""
    print(f"\n[5/5] Training for {EPOCHS} epochs...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Callbacks — special actions during training
    callbacks = [
        # Save the best model checkpoint automatically
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Stop early if model stops improving (saves time)
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,          # Stop if no improvement for 8 epochs
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # Train!
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on validation set
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  Final validation loss: {val_loss:.4f}")
    print(f"  Final validation MAE:  {val_mae:.4f}")
    print(f"  (MAE in duty units:    {val_mae * 255:.1f}/255)")

    # ── Convert to TFLite ──────────────────────────────────
    # TFLite = compressed format for running on Raspberry Pi
    print("\n  Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 quantization
    tflite_model = converter.convert()

    tflite_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  ✅ TFLite model saved: {tflite_path}  ({size_kb:.1f} KB)")

    # ── Save training metadata ─────────────────────────────
    meta = {
        "trained_at":   datetime.now().isoformat(),
        "epochs_run":   len(history.history["loss"]),
        "val_loss":     float(val_loss),
        "val_mae":      float(val_mae),
        "window_size":  WINDOW_SIZE,
        "training_rows":len(X_train),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return history, tflite_path


# ============================================================
#  MAIN
# ============================================================

def main():
    print("═"*55)
    print("  AETHER-FLOW AI MODEL TRAINER")
    print("═"*55)

    # Load data
    df = load_data()

    # Prepare features
    X_raw, y_raw, input_cols, output_cols = prepare_features(df)

    # Normalize to 0-1 range
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw)

    # Create sliding windows
    X_win, y_win = create_windows(X_scaled, y_scaled, WINDOW_SIZE)

    # Split into training and validation sets
    # 80% for training, 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_win, y_win, test_size=0.2, shuffle=False  # No shuffle — preserve time order
    )
    print(f"\n      Train samples: {len(X_train):,}")
    print(f"      Val samples:   {len(X_val):,}")

    # Build model
    model = build_model(WINDOW_SIZE, X_win.shape[2], y_win.shape[1])

    # Train and save
    history, tflite_path = train_and_save(model, X_train, y_train, X_val, y_val)

    print("\n" + "═"*55)
    print("  TRAINING COMPLETE!")
    print("═"*55)
    print(f"\n  Next steps:")
    print(f"  1. Copy {tflite_path}")
    print(f"     to your Raspberry Pi at:")
    print(f"     ./aetherflow_data/model/aetherflow_model.tflite")
    print(f"\n  2. In AetherFlow_DataLogger.py, set:")
    print(f"     AI_MODE = True")
    print(f"\n  3. Restart the data logger — it will now send")
    print(f"     AI commands to the ESP32!")
    print(f"\n  Watch for 'mode: AI' in the terminal output.")
    print("═"*55)


if __name__ == "__main__":
    main()
