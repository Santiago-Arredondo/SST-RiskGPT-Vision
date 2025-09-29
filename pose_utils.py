from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np

# Carga perezosa de MediaPipe para no romper si no está
try:
    import mediapipe as mp
    _MP_OK = True
    _POSE = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
except Exception:
    _MP_OK = False
    _POSE = None

def _angle(p1, p2, p3) -> float:
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    den = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    cosang = float(np.dot(v1, v2) / den)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def ergonomic_tokens(rgb_img: np.ndarray,
                     dets: List[Dict],
                     head_screen_delta_px: int = 40) -> Set[str]:
    """
    Devuelve tokens ergonómicos visibles: sitting_posture, bent_back, screen_below_eyes, laptop_on_lap.
    Si MediaPipe no está disponible, devuelve conjunto vacío.
    """
    toks: Set[str] = set()
    if not _MP_OK:
        return toks

    h, w = rgb_img.shape[:2]
    res = _POSE.process(rgb_img)
    if not res or not res.pose_landmarks:
        return toks

    lm = res.pose_landmarks.landmark

    def xy(idx): 
        pt = lm[idx]
        return (pt.x * w, pt.y * h)

    # Señales simples de postura sentada (aprox. por ángulos cadera/rodilla)
    hip_r = xy(24); knee_r = xy(26); shoulder_r = xy(12)
    try:
        angle_hip = _angle(shoulder_r, hip_r, knee_r)
        angle_knee = _angle(hip_r, knee_r, (knee_r[0], knee_r[1]+50))
        if angle_hip < 140 and angle_knee < 150:
            toks.add("sitting_posture")
    except Exception:
        pass

    # Flexión de tronco/cervical (ángulo bajo entre oído-hombro-cadera media)
    try:
        ear_r = xy(8)
        hip_mid = ((xy(23)[0]+xy(24)[0])/2.0, (xy(23)[1]+xy(24)[1])/2.0)
        trunk_angle = _angle(ear_r, shoulder_r, hip_mid)
        if trunk_angle < 150:
            toks.add("bent_back")
    except Exception:
        pass

    # Pantalla por debajo de ojos / laptop en regazo
    try:
        head_y = min(xy(8)[1], xy(7)[1])  # derecha/izquierda
        for d in dets:
            if d.get("cls") in {"screen", "monitor", "laptop"}:
                x1, y1, x2, y2 = d["box"]
                if (y1 - head_y) > head_screen_delta_px:
                    toks.add("screen_below_eyes")
                if "sitting_posture" in toks and (y2 > 0.75 * h) and d.get("cls") == "laptop":
                    toks.add("laptop_on_lap")
    except Exception:
        pass

    return toks
