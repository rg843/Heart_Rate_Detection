import time
from collections import deque
import numpy as np
import tensorflow as tf
import scipy.signal as signal
from filters import butter_bandpass_filter, OneDKalman


def _build_quality_model():
    # Build model with explicit Dense layers so layer indices are predictable
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    w0 = np.array([
        [1.2, -0.9, -0.8, 0.6],
        [0.8, 1.1, -1.1, 0.2],
        [0.5, -0.5, 1.0, 0.4],
        [1.0, 0.9, -0.5, -0.4],
        [0.4, 0.2, 0.9, 0.8],
        [-0.6, 0.8, 0.7, -0.2],
        [0.3, 0.3, 0.5, 1.0],
        [0.1, -0.4, 0.5, 0.8],
    ], dtype=np.float32)
    b0 = np.array([0.05] * 8, dtype=np.float32)
    w1 = np.array([[1.0], [0.9], [0.7], [0.3], [0.8], [0.5], [0.6], [0.4]], dtype=np.float32)
    b1 = np.array([-0.25], dtype=np.float32)
    # w0 is defined as (8,4); transpose to (4,8) to match Dense weight shape
    w0 = w0.T
    # Set weights for first and second Dense layers
    dense_layers = [L for L in model.layers if isinstance(L, tf.keras.layers.Dense)]
    if len(dense_layers) >= 2:
        dense_layers[0].set_weights([w0, b0])
        dense_layers[1].set_weights([w1, b1])
    else:
        # Fallback: try by index assuming Dense layers at 0 and 1
        model.layers[0].set_weights([w0, b0])
        model.layers[1].set_weights([w1, b1])
    return model


class SignalQualityModel:
    def __init__(self):
        self.model = _build_quality_model()

    def evaluate(self, features):
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        quality = float(self.model.predict(features, verbose=0)[0, 0])
        return float(np.clip(quality, 0.0, 1.0))


class EyeRPPGProcessor:
    def __init__(self, buffer_seconds=10, fps=25):
        self.buffer_seconds = buffer_seconds
        self.max_samples = int(buffer_seconds * fps)
        self.timestamps = []
        self.rgb_buffer = []
        self.motion_buffer = []
        self.quality_model = SignalQualityModel()
        self.kalman_bpm = OneDKalman(process_variance=1e-3, measurement_variance=0.3)
        self.kalman_accuracy = OneDKalman(process_variance=1e-3, measurement_variance=0.3)
        self.last_mean = None
        self.last_bpm = 0.0
        self.last_accuracy = 0.0
        self.last_quality = 0.0
        self.last_freq = None
        self.last_pulse = None
        self.last_status = "Initializing"
        # history buffers for UI plotting (store recent frequency and pulse values)
        self.freq_history = deque(maxlen=200)
        self.pulse_history = deque(maxlen=200)

    def reset(self):
        self.timestamps.clear()
        self.rgb_buffer.clear()
        self.motion_buffer.clear()
        self.last_mean = None
        self.freq_history.clear()
        self.pulse_history.clear()

    def add_sample(self, roi):
        now = time.time()
        mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)
        self.rgb_buffer.append(mean_rgb)
        self.timestamps.append(now)
        if self.last_mean is None:
            self.last_mean = mean_rgb
        motion = np.linalg.norm(mean_rgb - self.last_mean)
        self.motion_buffer.append(motion)
        self.last_mean = mean_rgb
        if len(self.rgb_buffer) > self.max_samples:
            self.rgb_buffer.pop(0)
            self.timestamps.pop(0)
            self.motion_buffer.pop(0)

    def estimate(self):
        if len(self.timestamps) < 60:
            return None
        times = np.array(self.timestamps)
        dt = np.maximum(np.mean(np.diff(times)), 1e-3)
        fs = 1.0 / dt
        if fs < 10:
            fs = 25.0
        rgb = np.array(self.rgb_buffer)
        green = rgb[:, 1]
        green_centered = green - np.mean(green)
        filtered = butter_bandpass_filter(green_centered, 0.7, 3.5, fs, order=4)
        if len(filtered) < 32:
            return None
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        fftmag = np.abs(np.fft.rfft(filtered))
        valid = (freqs >= 0.7) & (freqs <= 3.5)
        if not np.any(valid):
            return None
        selected = freqs[valid]
        selected_mag = fftmag[valid]
        peak_idx = int(np.argmax(selected_mag))
        freq_hz = float(selected[peak_idx])
        bpm = float(freq_hz * 60.0)
        quality = self._compute_quality(selected_mag, filtered, np.mean(self.motion_buffer[-n:]))
        accuracy = 10.0 + 85.0 * quality
        self.last_bpm = float(np.clip(self.kalman_bpm.update(bpm), 30.0, 180.0))
        self.last_accuracy = float(np.clip(self.kalman_accuracy.update(accuracy / 100.0) * 100.0, 0.0, 100.0))
        self.last_quality = quality
        self.last_status = self._assess_status(self.last_bpm, quality)
        # frequency trend detection
        trend = "stable"
        if self.last_freq is None:
            trend = "stable"
        else:
            delta = freq_hz - self.last_freq
            if delta > 0.015:
                trend = "up"
            elif delta < -0.015:
                trend = "down"
            else:
                trend = "stable"
        self.last_freq = freq_hz
        # compute pulse amplitude (envelope) and peak-to-peak as pulse metric
        try:
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)
            pulse_mean = float(np.mean(envelope))
            pulse_p2p = float(np.max(envelope) - np.min(envelope))
        except Exception:
            pulse_mean = 0.0
            pulse_p2p = 0.0

        # pulse trend detection (relative change)
        pulse_trend = "stable"
        if self.last_pulse is None:
            pulse_trend = "stable"
        else:
            if self.last_pulse > 1e-6:
                rel = (pulse_mean - self.last_pulse) / (self.last_pulse)
                if rel > 0.05:
                    pulse_trend = "up"
                elif rel < -0.05:
                    pulse_trend = "down"
                else:
                    pulse_trend = "stable"
        self.last_pulse = pulse_mean

        # append recent metrics to histories for plotting (guarded)
        try:
            self.freq_history.append(float(freq_hz))
        except Exception:
            pass

        try:
            self.pulse_history.append(float(pulse_mean))
        except Exception:
            pass

        return {
            "bpm": self.last_bpm,
            "accuracy": self.last_accuracy,
            "quality": self.last_quality,
            "status": self.last_status,
            "frequency_hz": float(freq_hz),
            "frequency_trend": trend,
            "pulse": float(pulse_mean),
            "pulse_p2p": float(pulse_p2p),
            "pulse_trend": pulse_trend,
        }

    def _compute_quality(self, spectrum, filtered_signal, motion_level):
        peak_power = float(np.max(spectrum)) if spectrum.size else 0.0
        total_power = float(np.sum(spectrum)) + 1e-6
        peak_ratio = peak_power / total_power
        stability = 1.0 - min(1.0, np.std(filtered_signal) / (np.mean(np.abs(filtered_signal)) + 1e-3))
        motion_score = 1.0 - np.tanh(motion_level / 20.0)
        features = [peak_ratio, stability, motion_score, 0.5]
        quality = self.quality_model.evaluate(features)
        return quality

    def _assess_status(self, bpm, quality):
        if quality < 0.30:
            return "Signal Weak"
        if bpm < 55:
            return "Low"
        if bpm <= 100:
            return "Normal"
        return "High"
