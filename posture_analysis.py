def analyze_posture(video_path):
    import cv2
    import mediapipe as mp
    import numpy as np

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    movement_scores = []
    prev_landmarks = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            current_landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

            if prev_landmarks:
                diffs = [np.linalg.norm(np.array(curr) - np.array(prev)) for curr, prev in zip(current_landmarks, prev_landmarks)]
                movement_scores.append(np.mean(diffs))

            prev_landmarks = current_landmarks

    cap.release()
    avg_movement = np.mean(movement_scores) if movement_scores else 0

    # Kullanıcıya sade geri bildirim
    if avg_movement < 0.001:
        feedback = "Sunum boyunca neredeyse hiç hareket etmedin. Biraz daha doğal hareketler kullanabilirsin."
    elif avg_movement > 0.01:
        feedback = "Sunum sırasında oldukça hareketliydin. Gereğinden fazla kıpırdanmak dikkat dağıtabilir."
    else:
        feedback = "Beden dilin dengeliydi. Etkili bir duruş sergiledin."

    return {
        "posture_feedback": feedback  # sadece bunu frontend'e göstereceğiz
    }
