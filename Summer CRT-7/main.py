# Clean imports
import time
import sys
import cv2
import numpy as np
from processor import EyeRPPGProcessor
from ui_utils import draw_futuristic_overlay, draw_dashboard_with_charts as draw_dashboard

# Try to import MediaPipe Face Mesh. If unavailable or incompatible, fall back
# to OpenCV Haar cascades for face+eye detection.
USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        USE_MEDIAPIPE = True
except Exception:
    USE_MEDIAPIPE = False

if not USE_MEDIAPIPE:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153, 154]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 373, 380, 381, 382, 362]


def landmark_points(landmarks, image_w, image_h, indices):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * image_w), int(lm.y * image_h)))
    return np.array(pts, dtype=np.int32)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    return x1, y1, x2, y2


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    processor = EyeRPPGProcessor(buffer_seconds=10, fps=30)
    face_mesh = None
    if USE_MEDIAPIPE:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    last_timestamp = time.time()
    fps = 0.0
    # snapshot mode: save one frame with charts and exit when `--snapshot` passed
    snapshot_mode = '--snapshot' in sys.argv
    last_snapshot_time = 0.0
    snapshot_interval = 2.5
    snapshot_path = 'snapshot.png'

    try:
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break
            metrics = None
            bpm = 0.0
            accuracy = 0.0
            quality = 0.0
            status = "Waiting"
            quality_text = "INIT"
            frequency_hz = 0.0
            freq_trend = "stable"

            h, w = frame.shape[:2]

            if USE_MEDIAPIPE and face_mesh is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
                image.flags.writeable = True
                if results and getattr(results, 'multi_face_landmarks', None):
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    left_pts = landmark_points(face_landmarks, w, h, LEFT_EYE_INDICES)
                    right_pts = landmark_points(face_landmarks, w, h, RIGHT_EYE_INDICES)
                    frame = draw_futuristic_overlay(frame, left_pts, right_pts)
                    lx1, ly1, lx2, ly2 = cv2.boundingRect(left_pts)
                    rx1, ry1, rx2, ry2 = cv2.boundingRect(right_pts)
                    x1 = min(lx1, rx1) - 8
                    y1 = min(ly1, ry1) - 6
                    x2 = max(lx1 + lx2, rx1 + rx2) + 8
                    y2 = max(ly1 + ly2, ry1 + ry2) + 6
                    x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), w, h)
                    if x2 > x1 and y2 > y1:
                        eye_roi = frame[y1:y2, x1:x2]
                        if eye_roi.size:
                            processor.add_sample(eye_roi)
                            metrics = processor.estimate()
                else:
                    processor.reset()
            else:
                # Haar cascade fallback
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
                if len(faces) > 0:
                    (fx, fy, fw, fh) = faces[0]
                    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
                    face_roi_color = frame[fy:fy+fh, fx:fx+fw]
                    eyes = eye_cascade.detectMultiScale(face_roi_gray)
                    if len(eyes) >= 1:
                        # take leftmost and rightmost if available
                        eyes_sorted = sorted(eyes, key=lambda e: e[0])
                        if len(eyes_sorted) == 1:
                            ex, ey, ew, eh = eyes_sorted[0]
                            ex += fx; ey += fy
                            x1 = ex - 8; y1 = ey - 6; x2 = ex + ew + 8; y2 = ey + eh + 6
                        else:
                            ex1, ey1, ew1, eh1 = eyes_sorted[0]
                            ex2, ey2, ew2, eh2 = eyes_sorted[-1]
                            ex1 += fx; ey1 += fy
                            ex2 += fx; ey2 += fy
                            x1 = min(ex1, ex2) - 8
                            y1 = min(ey1, ey2) - 6
                            x2 = max(ex1 + ew1, ex2 + ew2) + 8
                            y2 = max(ey1 + eh1, ey2 + eh2) + 6
                        x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), w, h)
                        if x2 > x1 and y2 > y1:
                            eye_roi = frame[y1:y2, x1:x2]
                            if eye_roi.size:
                                processor.add_sample(eye_roi)
                                metrics = processor.estimate()
                        # draw simple eye overlays
                        left_overlay = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
                        frame = draw_futuristic_overlay(frame, left_overlay, left_overlay)
                    else:
                        processor.reset()
                else:
                    processor.reset()

            # Process metrics and UI
            if metrics:
                bpm = metrics["bpm"]
                accuracy = metrics["accuracy"]
                quality = metrics["quality"]
                status = metrics["status"]
                frequency_hz = metrics.get("frequency_hz", 0.0)
                freq_trend = metrics.get("frequency_trend", "stable")
                pulse = metrics.get("pulse", 0.0)
                pulse_p2p = metrics.get("pulse_p2p", 0.0)
                pulse_trend = metrics.get("pulse_trend", "stable")
                quality_text = "GOOD" if quality >= 0.6 else "STABLE" if quality >= 0.4 else "LOW"
                print(f"{time.strftime('%H:%M:%S')}  BPM={bpm:05.1f}  FREQ={frequency_hz:0.2f}Hz  TREND={freq_trend}  PULSE={pulse:0.4f} {pulse_trend}  ACC={accuracy:05.1f}%  Q={quality:0.3f}  STATUS={status}", flush=True)

            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, time.time() - last_timestamp))
            last_timestamp = time.time()
            frame = draw_dashboard(frame, bpm, accuracy, quality_text, status, fps, frequency_hz, freq_trend, pulse if metrics else 0.0, pulse_trend if metrics else 'stable', processor.freq_history, processor.pulse_history)
            # if in snapshot mode, save a frame with the dashboard/chart and exit (even without metrics)
            if snapshot_mode and (time.time() - last_snapshot_time) > snapshot_interval:
                try:
                    cv2.imwrite(snapshot_path, frame)
                    print(f"Saved snapshot to {snapshot_path}", flush=True)
                except Exception as e:
                    print(f"Snapshot save failed: {e}", flush=True)
                break
            cv2.imshow("AI Heart Rate Monitor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if face_mesh is not None:
            try:
                face_mesh.close()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
 
