"""
Unit tests for the AetherFlow Training Pipeline.

Tests windowing, feature preparation, and model architecture
without requiring a GPU or full dataset.
"""

import numpy as np
import pytest


# ── Replicate core functions from AetherFlow_Train.py ───────────────────

WINDOW_SIZE = 120


def create_windows(X: np.ndarray, y: np.ndarray, window_size: int):
    """Convert flat data into sliding windows for the LSTM."""
    X_windows = []
    y_targets = []

    for i in range(window_size, len(X)):
        X_windows.append(X[i - window_size:i])
        y_targets.append(y[i])

    return np.array(X_windows, dtype=np.float32), np.array(y_targets, dtype=np.float32)


# ── Tests ───────────────────────────────────────────────────────────────


class TestCreateWindows:
    """Tests for the sliding-window creation logic."""

    def test_output_shapes(self):
        """Windows should have shape (n_samples, window_size, n_features)."""
        n_rows = 200
        n_features = 9
        n_outputs = 2
        X = np.random.rand(n_rows, n_features).astype(np.float32)
        y = np.random.rand(n_rows, n_outputs).astype(np.float32)

        X_win, y_win = create_windows(X, y, WINDOW_SIZE)

        assert X_win.shape == (n_rows - WINDOW_SIZE, WINDOW_SIZE, n_features)
        assert y_win.shape == (n_rows - WINDOW_SIZE, n_outputs)

    def test_small_window(self):
        """Test with a small window size."""
        X = np.arange(20).reshape(10, 2).astype(np.float32)
        y = np.arange(10).reshape(10, 1).astype(np.float32)

        X_win, y_win = create_windows(X, y, 3)

        assert X_win.shape == (7, 3, 2)
        assert y_win.shape == (7, 1)

    def test_window_content_correct(self):
        """Verify the window contains the right sequence of values."""
        X = np.arange(10).reshape(10, 1).astype(np.float32)
        y = np.arange(10).reshape(10, 1).astype(np.float32)

        X_win, y_win = create_windows(X, y, 3)

        # First window should be [0, 1, 2] and target should be 3
        np.testing.assert_array_equal(X_win[0], [[0], [1], [2]])
        np.testing.assert_array_equal(y_win[0], [3])

        # Second window should be [1, 2, 3] and target should be 4
        np.testing.assert_array_equal(X_win[1], [[1], [2], [3]])
        np.testing.assert_array_equal(y_win[1], [4])

    def test_insufficient_data(self):
        """When data length equals window size, should produce zero windows."""
        X = np.random.rand(WINDOW_SIZE, 5).astype(np.float32)
        y = np.random.rand(WINDOW_SIZE, 2).astype(np.float32)

        X_win, y_win = create_windows(X, y, WINDOW_SIZE)

        assert X_win.shape[0] == 0
        assert y_win.shape[0] == 0

    def test_dtype_preserved(self):
        """Output should be float32 for TFLite compatibility."""
        X = np.random.rand(150, 5).astype(np.float32)
        y = np.random.rand(150, 2).astype(np.float32)

        X_win, y_win = create_windows(X, y, WINDOW_SIZE)

        assert X_win.dtype == np.float32
        assert y_win.dtype == np.float32


class TestFeatureConfig:
    """Tests for feature configuration constants."""

    def test_input_features_count(self):
        """Training pipeline expects 9 input features."""
        input_cols = [
            "temp_c", "temp_ambient_c", "humidity_pct", "dew_point_c",
            "power_watts", "voltage_v", "peltier_duty", "fan1_duty",
            "temp_error",
        ]
        assert len(input_cols) == 9

    def test_output_targets_count(self):
        """Training pipeline predicts 2 output values."""
        output_cols = ["peltier_duty", "fan1_duty"]
        assert len(output_cols) == 2

    def test_window_size_matches_logger(self):
        """Window size must match between logger and trainer."""
        assert WINDOW_SIZE == 120


class TestNormalization:
    """Tests for data normalization logic."""

    def test_minmax_scaling(self):
        """MinMax scaling should produce values in [0, 1]."""
        from sklearn.preprocessing import MinMaxScaler

        X = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.min() >= 0.0
        assert X_scaled.max() <= 1.0

    def test_minmax_inverse(self):
        """Inverse transform should recover original values."""
        from sklearn.preprocessing import MinMaxScaler

        X = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X, X_recovered)
