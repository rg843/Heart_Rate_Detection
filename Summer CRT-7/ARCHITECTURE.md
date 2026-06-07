# Project Architecture

## Module Breakdown

- `main.py`
  - Captures webcam frames using OpenCV
  - Uses MediaPipe Face Mesh to locate face landmarks
  - Extracts left and right eye regions for rPPG analysis
  - Sends eye ROI samples to the processing pipeline
  - Draws the futuristic UI and live metrics overlay

- `processor.py`
  - Maintains a buffer of eye-region RGB samples
  - Applies a Butterworth bandpass filter to remove noise
  - Uses FFT peak detection to estimate heart rate (BPM)
  - Uses a lightweight TensorFlow model as signal quality evaluator
  - Smoothes outputs with a 1D Kalman filter

- `filters.py`
  - Implements the bandpass filtering pipeline
  - Provides a reusable one-dimensional Kalman filter class

- `ui_utils.py`
  - Draws neon green eye overlays on the camera feed
  - Renders a dark, futuristic dashboard
  - Adds heartbeat animation and polished telemetry

## Signal Flow

1. `main.py` captures a frame and detects facial landmarks.
2. Eye landmarks are converted into pixel regions for both eyes.
3. The combined eye ROI is averaged to extract an RGB signal.
4. `processor.py` buffers signal samples for a 10-second window.
5. The buffered green-channel waveform is filtered to the pulse band.
6. FFT analysis estimates the dominant pulse frequency and BPM.
7. TensorFlow model converts spectral and motion features into a quality score.
8. Kalman smoothing produces stable BPM and accuracy outputs.
9. `ui_utils.py` renders the final metrics, status, and animation.

## Presentation-Ready Flow
- Input: webcam frames
- Detection: MediaPipe Face Mesh
- Processing: rPPG -> filter -> FFT -> Kalman -> TensorFlow scoring
- Output: neon dashboard with BPM, accuracy, signal quality, and health status
