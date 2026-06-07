import numpy as np
import cv2
from ui_utils import draw_dashboard_with_charts

def make_blank(w=1280, h=720, color=(6, 8, 10)):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame

def simulate_histories():
    t = np.linspace(0, 1.0, 120)
    freq = 1.0 + 0.2 * np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(t.size)
    pulse = 0.7 + 0.15 * np.sin(2 * np.pi * 2.5 * t) + 0.03 * np.random.randn(t.size)
    return list(freq.tolist()), list(pulse.tolist())

def main():
    frame = make_blank()
    freq_hist, pulse_hist = simulate_histories()
    out = draw_dashboard_with_charts(frame, bpm=72, accuracy=90.5, quality_text='GOOD', status='Normal', fps=29.7, frequency_hz=1.12, freq_trend='stable', pulse=0.72, pulse_trend='stable', freq_history=freq_hist, pulse_history=pulse_hist)
    path = 'snapshot.png'
    cv2.imwrite(path, out)
    print('Wrote', path)

if __name__ == '__main__':
    main()
