import cv2
import mediapipe as mp
import math
import time
import threading
import winsound

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
MOUTH = [13, 14, 78, 308] 

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_ear(eye, lm, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in eye]
    v1, v2, h1 = dist(pts[1], pts[5]), dist(pts[2], pts[4]), dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * h1) if h1 != 0 else 0

def get_mar(mouth, lm, w, h):
    """Рахує Mouth Aspect Ratio для позіхання"""
    v = dist((lm[mouth[0]].x * w, lm[mouth[0]].y * h), (lm[mouth[1]].x * w, lm[mouth[1]].y * h))
    h_dist = dist((lm[mouth[2]].x * w, lm[mouth[2]].y * h), (lm[mouth[3]].x * w, lm[mouth[3]].y * h))
    return v / h_dist if h_dist != 0 else 0

def get_iris_center(iris, lm, w, h):
    cx = sum([lm[i].x for i in iris]) / len(iris) * w
    cy = sum([lm[i].y for i in iris]) / len(iris) * h
    return (cx, cy)

alarm_state = "NONE"

def sound_daemon():
    global alarm_state
    while True:
        if alarm_state == "DISTRACTED":
            winsound.Beep(1200, 150)
            winsound.Beep(800, 150)
        elif alarm_state == "MICROSLEEP":
            winsound.Beep(2500, 400) 
        else:
            time.sleep(0.05)

threading.Thread(target=sound_daemon, daemon=True).start()

blink_count, is_eye_closed = 0, False
close_time, last_blink_end_time = 0, None
blink_durations, intervals, yawns_count = [], [], 0

total_time = 0
distracted_time = 0
tracking_start_time = None

calibration_frames = 0
REQUIRED_CALIB_FRAMES = 50
calib_data = {"ear": [], "mar": [], "yaw": [], "pitch": [], "gaze_x": [], "gaze_y": []}
base_metrics = {}

print("СИСТЕМА ЗАПУЩЕНА!")
cap = cv2.VideoCapture(0)

while cap.isOpened() and blink_count < 5:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_alarm = "NONE"
    distracted_reason = ""
    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True
        lm = results.multi_face_landmarks[0].landmark
        
        left_ear = get_ear(LEFT_EYE, lm, w, h)
        right_ear = get_ear(RIGHT_EYE, lm, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = get_mar(MOUTH, lm, w, h)

        nose = lm[1]
        l_cheek, r_cheek = lm[234], lm[454]
        top_head, chin = lm[10], lm[152]
        
        yaw = abs(nose.x - l_cheek.x) / (abs(r_cheek.x - l_cheek.x) + 0.001)
        pitch = abs(nose.y - top_head.y) / (abs(chin.y - top_head.y) + 0.001)

        l_iris = get_iris_center(LEFT_IRIS, lm, w, h)
        l_inner = (lm[362].x * w, lm[362].y * h)
        l_outer = (lm[263].x * w, lm[263].y * h)
        l_top = (lm[386].x * w, lm[386].y * h)
        l_bottom = (lm[374].x * w, lm[374].y * h)
        
        gaze_x = abs(l_iris[0] - l_inner[0]) / (abs(l_outer[0] - l_inner[0]) + 0.001)
        gaze_y = abs(l_iris[1] - l_top[1]) / (abs(l_bottom[1] - l_top[1]) + 0.001)

        if calibration_frames < REQUIRED_CALIB_FRAMES:
            cv2.putText(frame, f"CALIBRATING... LOOK STRAIGHT ({calibration_frames}/{REQUIRED_CALIB_FRAMES})", 
                        (20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            if avg_ear > 0.2: 
                calib_data["ear"].append(avg_ear)
                calib_data["mar"].append(mar)
                calib_data["yaw"].append(yaw)
                calib_data["pitch"].append(pitch)
                calib_data["gaze_x"].append(gaze_x)
                calib_data["gaze_y"].append(gaze_y)
                calibration_frames += 1
            
            if calibration_frames == REQUIRED_CALIB_FRAMES:
                base_metrics["ear"] = sum(calib_data["ear"]) / len(calib_data["ear"]) * 0.65 
                base_metrics["mar"] = sum(calib_data["mar"]) / len(calib_data["mar"]) * 2.5  
                base_metrics["yaw"] = sum(calib_data["yaw"]) / len(calib_data["yaw"])
                base_metrics["pitch"] = sum(calib_data["pitch"]) / len(calib_data["pitch"])
                base_metrics["gaze_x"] = sum(calib_data["gaze_x"]) / len(calib_data["gaze_x"])
                base_metrics["gaze_y"] = sum(calib_data["gaze_y"]) / len(calib_data["gaze_y"])
                tracking_start_time = time.time()
                print("\nКалібрування завершено! Пороги встановлено автоматично.")

        else:
            if avg_ear < base_metrics["ear"]:
                if not is_eye_closed:
                    is_eye_closed = True
                    close_time = time.time()
                    if last_blink_end_time: intervals.append(close_time - last_blink_end_time)
                else:
                    if time.time() - close_time > 1.5:
                        current_alarm = "MICROSLEEP"
                        distracted_reason = "WAKE UP!"
            else:
                if is_eye_closed:
                    is_eye_closed = False
                    duration = time.time() - close_time
                    if duration > 0.05: 
                        blink_count += 1
                        blink_durations.append(duration)
                        last_blink_end_time = time.time()
            
            if mar > base_metrics["mar"]:
                distracted_reason = "YAWNING"
                yawns_count += 1 
            
            if not is_eye_closed and current_alarm != "MICROSLEEP":
                dev_gaze_x = abs(gaze_x - base_metrics["gaze_x"])
                dev_gaze_y = abs(gaze_y - base_metrics["gaze_y"])
                dev_yaw = abs(yaw - base_metrics["yaw"])
                dev_pitch = abs(pitch - base_metrics["pitch"])

                if dev_yaw > 0.12 or dev_pitch > 0.12:
                    current_alarm = "DISTRACTED"
                    distracted_reason = "HEAD TURNED"
                elif dev_gaze_x > 0.18 or dev_gaze_y > 0.20:
                    current_alarm = "DISTRACTED"
                    distracted_reason = "LOOKING AWAY"

            cv2.putText(frame, f"Blinks: {blink_count}/5", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Status: {distracted_reason if distracted_reason else 'FOCUSED'}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if current_alarm != "NONE" else (0, 255, 0), 2)

    if not face_detected and calibration_frames == REQUIRED_CALIB_FRAMES:
        current_alarm = "DISTRACTED"
        cv2.putText(frame, "NO FACE DETECTED", (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    alarm_state = current_alarm
    if tracking_start_time and current_alarm != "NONE":
        distracted_time += 1/30 

    cv2.imshow('Cyberpunk Pro Tracker', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

alarm_state = "NONE" 
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("КІБЕРПАНК ЗВІТ:")
print("="*50)

if tracking_start_time:
    total_time = time.time() - tracking_start_time
    focus_score = max(0, 100 - (distracted_time / total_time * 100))
    
    print(f"Рейтинг уважності (Focus Score): {focus_score:.1f}%")
    if yawns_count > 10: 
        print(f"Виявлено ознаки втоми (позіхання).")

if blink_count == 5:
    avg_dur = sum(blink_durations) / len(blink_durations)
    avg_int = sum(intervals) / len(intervals) if intervals else 0
    print(f"Кліпань: 5/5")
    print(f"Сер. тривалість: {avg_dur:.3f}с | Сер. проміжок: {avg_int:.3f}с")
else:
    print(f"Перервано юзером. Кліпань: {blink_count}/5")
print("="*50)
