import time
import cv2
import numpy as np
import math


def draw_futuristic_overlay(frame, left_eye_pts, right_eye_pts):
    overlay = frame.copy()
    glow_color = (32, 255, 90)
    line_color = (30, 220, 70)
    thickness = 2
    cv2.polylines(overlay, [left_eye_pts], True, glow_color, thickness * 4, cv2.LINE_AA)
    cv2.polylines(overlay, [right_eye_pts], True, glow_color, thickness * 4, cv2.LINE_AA)
    cv2.polylines(overlay, [left_eye_pts], True, line_color, thickness, cv2.LINE_AA)
    cv2.polylines(overlay, [right_eye_pts], True, line_color, thickness, cv2.LINE_AA)
    for pts in (left_eye_pts, right_eye_pts):
        for x, y in pts:
            cv2.circle(overlay, (x, y), 2, line_color, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)


def draw_dashboard(frame, bpm, accuracy, quality_text, status, fps, frequency_hz=0.0, freq_trend='stable', pulse=0.0, pulse_trend='stable'):
    h, w = frame.shape[:2]
    panel_w = min(360, int(w * 0.35))
    overlay = frame.copy()
    start_x = w - panel_w
    cv2.rectangle(overlay, (start_x, 0), (w, h), (12, 18, 20), -1)
    alpha = 0.72
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    title_x = start_x + 24
    cv2.putText(frame, "NEO-HEART AI", (title_x, 48), cv2.FONT_HERSHEY_DUPLEX, 1.05, (128, 255, 150), 2, cv2.LINE_AA)
    cv2.putText(frame, "REAL-TIME BIOMETRICS", (title_x, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 190), 1, cv2.LINE_AA)
    cv2.line(frame, (title_x, 86), (w - 18, 86), (40, 180, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, f"BPM: {int(bpm):03d}", (title_x, 132), cv2.FONT_HERSHEY_DUPLEX, 1.6, (24, 255, 112), 3, cv2.LINE_AA)
    arrow = '→'
    if freq_trend == 'up':
        arrow = '↑'
    elif freq_trend == 'down':
        arrow = '↓'
    cv2.putText(frame, f"FREQ: {frequency_hz:0.2f}Hz {arrow}", (title_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200, 255, 180), 2, cv2.LINE_AA)
    # Pulse amplitude display
    p_arrow = '→'
    if pulse_trend == 'up':
        p_arrow = '↑'
    elif pulse_trend == 'down':
        p_arrow = '↓'
    cv2.putText(frame, f"PULSE: {pulse:0.4f} {p_arrow}", (title_x, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (170, 255, 170), 2, cv2.LINE_AA)
    cv2.putText(frame, f"ACC: {accuracy:05.1f}%", (title_x, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 255, 190), 2, cv2.LINE_AA)
    cv2.putText(frame, f"QTY: {quality_text}", (title_x, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 255, 170), 2, cv2.LINE_AA)
    cv2.putText(frame, f"STATUS: {status}", (title_x, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 140), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (title_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 255, 190), 1, cv2.LINE_AA)
    heart_center = (start_x + panel_w - 60, 160)
    draw_heartbeat(frame, heart_center, int(time.time() * 4) % 2 == 0)
    return frame


def draw_signal_chart(frame, history, rect, color=(32, 255, 90)):
    if history is None:
        return
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (8, 12, 10), -1)
    arr = np.array(list(history), dtype=np.float32)
    if arr.size < 2:
        return
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if abs(mx - mn) < 1e-6:
        mx = mn + 1e-6
    norm = (arr - mn) / (mx - mn)
    pts = []
    ln = len(norm)
    for i, v in enumerate(norm):
        px = int(x + (i / (ln - 1)) * w)
        py = int(y + (1.0 - v) * h)
        pts.append((px, py))
    cv2.polylines(frame, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)


def draw_dashboard_with_charts(frame, bpm, accuracy, quality_text, status, fps, frequency_hz=0.0, freq_trend='stable', pulse=0.0, pulse_trend='stable', freq_history=None, pulse_history=None):
    # reuse existing dashboard then draw small sparklines for frequency and pulse
    frame = draw_dashboard(frame, bpm, accuracy, quality_text, status, fps, frequency_hz, freq_trend, pulse, pulse_trend)
    h, w = frame.shape[:2]
    panel_w = min(360, int(w * 0.35))
    start_x = w - panel_w
    chart_w = panel_w - 48
    # frequency chart
    freq_rect = (start_x + 24, 300, chart_w, 56)
    draw_signal_chart(frame, freq_history, freq_rect, color=(32, 235, 120))
    cv2.putText(frame, 'Freq (Hz)', (freq_rect[0], freq_rect[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
    # pulse chart
    pulse_rect = (start_x + 24, 364, chart_w, 56)
    draw_signal_chart(frame, pulse_history, pulse_rect, color=(180, 255, 140))
    cv2.putText(frame, 'Pulse (env)', (pulse_rect[0], pulse_rect[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
    return frame


def draw_heartbeat(frame, center, active):
    x, y = center
    size = 16 if active else 12
    color = (18, 255, 140)
    thickness = 2
    points = np.array([
        (x, y),
        (x - size, y - size),
        (x - size * 2, y),
        (x, y + size * 2),
        (x + size * 2, y),
        (x + size, y - size),
    ], np.int32)
    cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)
    if active:
        cv2.circle(frame, (x, y), size // 2, color, -1, cv2.LINE_AA)


def make_text_color(quality):
    if quality >= 0.75:
        return (20, 255, 110)
    if quality >= 0.45:
        return (220, 235, 90)
    return (10, 180, 220)
