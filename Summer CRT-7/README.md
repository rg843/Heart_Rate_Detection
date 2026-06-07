# AI-Powered Real-Time Heart Rate Detection System

A professional, futuristic Python application designed for real-time heart rate estimation using webcam eye-region analysis, MediaPipe face mesh, remote photoplethysmography (rPPG), Butterworth bandpass filtering, Kalman smoothing, and TensorFlow-based signal quality evaluation.

## Features
- Real-time webcam face and eye detection with MediaPipe Face Mesh
- Neon green eye overlays and live futuristic dashboard
- Eye-region RGB signal capture for rPPG heart rate estimation
- Butterworth bandpass filtering for noise suppression
- Kalman filter smoothing for stable BPM output
- TensorFlow-powered signal quality and accuracy scoring
- Health status detection: Normal / Low / High / Signal Weak
- Dark futuristic interface with heartbeat animation
- Modular design for project presentation and expansion

## Project Structure
- `main.py` - Real-time capture, face and eye tracking, UI rendering
- `processor.py` - rPPG processing, FFT-based BPM estimation, quality evaluation
- `filters.py` - Butterworth filter and Kalman filter implementations
- `ui_utils.py` - Dashboard and eye-overlay drawing utilities
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   - Windows:
     ```powershell
     .\\venv\\Scripts\\Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the application with:
```bash
python main.py
```

Make sure your webcam is connected and visible. Align your face so both eyes are in the camera view. The dashboard displays BPM, signal quality, accuracy, and health status.

## Architecture Overview
The application is built as a real-time signal processing pipeline:
1. Webcam frames are captured with OpenCV.
2. MediaPipe Face Mesh tracks facial landmarks and extracts eye regions.
3. Average RGB values from the eye ROI are buffered over time.
4. A Butterworth bandpass filter isolates pulse frequencies between 0.7 and 3.5 Hz.
5. FFT peak detection converts the filtered signal into BPM.
6. A Kalman filter smooths noisy BPM and accuracy values.
7. A TensorFlow-based quality evaluator scores signal reliability.
8. The UI layer renders a neon dashboard and animated heartbeat.

## Notes for Presentation
- The system is designed for a minimum 20 FPS experience.
- Eye-region capture improves robustness to head movement and lighting changes.
- The TensorFlow model is used as a lightweight quality estimator, making the system AI-powered.
- The modular architecture supports future upgrades: training a quality model, multi-person tracking, or cloud-based monitoring.

## Future Improvements
- Add a trained TensorFlow model for more precise quality and health prediction.
- Support multiple faces and per-person heart rate display.
- Add data logging and export to CSV for research validation.
- Integrate more advanced rPPG methods like CHROM or POS.

## License
This project is intended for academic and presentation use.
