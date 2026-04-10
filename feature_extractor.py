# feature_extractor.py
import numpy as np

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C + 1e-6)

def get_iris_center(landmarks, iris_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices]
    cx = np.mean([p[0] for p in pts])
    cy = np.mean([p[1] for p in pts])
    return cx, cy

def extract_features(face_landmarks, w, h):
    lm = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks

    lx, ly = get_iris_center(lm, LEFT_IRIS,  w, h)
    rx, ry = get_iris_center(lm, RIGHT_IRIS, w, h)

    gaze_x = (lx + rx) / 2
    gaze_y = (ly + ry) / 2

    le_pos = lm[LEFT_EYE[0]]
    re_pos = lm[RIGHT_EYE[0]]

    left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
    right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
    pupil_left  = left_ear  * 5.0
    pupil_right = right_ear * 5.0

    le_cx = np.mean([lm[i].x for i in LEFT_EYE])  * w
    le_cy = np.mean([lm[i].y for i in LEFT_EYE])  * h
    re_cx = np.mean([lm[i].x for i in RIGHT_EYE]) * w
    re_cy = np.mean([lm[i].y for i in RIGHT_EYE]) * h

    gaze_dir_lx = (lx - le_cx) / (w + 1e-6)
    gaze_dir_ly = (ly - le_cy) / (h + 1e-6)
    gaze_dir_rx = (rx - re_cx) / (w + 1e-6)
    gaze_dir_ry = (ry - re_cy) / (h + 1e-6)

    features = [
        gaze_x, gaze_y,
        lx, ly,
        rx, ry,
        gaze_dir_lx, gaze_dir_ly, 0.0,
        gaze_dir_rx, gaze_dir_ry, 0.0,
        pupil_left, pupil_right,
        le_pos.x * w, le_pos.y * h, le_pos.z,
        re_pos.x * w, re_pos.y * h, re_pos.z,
        lx, ly,
        rx, ry,
        gaze_x / w, gaze_y / h,
        lx / w, ly / h,
        rx / w, ry / h,
        left_ear * 100,
        gaze_x, gaze_y,
        gaze_x / w, gaze_y / h,
        0.0,
        (pupil_left + pupil_right) / 2,
    ]

    if len(features) < 38:
        features.extend([0.0] * (38 - len(features)))

    return np.array(features[:38], dtype='float32')