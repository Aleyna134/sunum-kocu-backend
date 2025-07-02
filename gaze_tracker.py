import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def is_looking_forward(landmarks, image_width):
    left_eye = landmarks[33]  # Sol göz bebeği
    right_eye = landmarks[263]  # Sağ göz bebeği

    eye_center_x = (left_eye.x + right_eye.x) / 2
    face_center = 0.5

    deviation = abs(eye_center_x - face_center)
    return deviation < 0.05  # Eğer merkezden sapma azsa, kameraya bakıyor say

def analyze_gaze(video_path):
    cap = cv2.VideoCapture(video_path)
    gaze_count = 0
    looking = False
    look_start = 0
    total_look_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            if is_looking_forward(landmarks, frame.shape[1]):
                if not looking:
                    gaze_count += 1
                    look_start = time.time()
                    looking = True
            else:
                if looking:
                    total_look_time += time.time() - look_start
                    looking = False
        else:
            if looking:
                total_look_time += time.time() - look_start
                looking = False

    cap.release()
    face_mesh.close()

    avg_duration = total_look_time / gaze_count if gaze_count > 0 else 0

    return {
        "gaze_count": gaze_count,
        "total_gaze_time_sec": round(avg_duration, 2)
    }
