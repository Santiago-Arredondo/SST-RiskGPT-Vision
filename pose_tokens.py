# pose_tokens.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Set
import math

KP = {"nose":0,"l_shoulder":5,"r_shoulder":6,"l_hip":11,"r_hip":12}

def _ang(v1, v2):
    def norm(v):
        n = math.hypot(v[0], v[1]); 
        return (v[0]/(n+1e-6), v[1]/(n+1e-6))
    a, b = norm(v1), norm(v2)
    dot = max(-1.0, min(1.0, a[0]*b[0] + a[1]*b[1]))
    return math.degrees(math.acos(dot))

def _vec(a, b): return (b[0]-a[0], b[1]-a[1])
def _mid(a, b): return ((a[0]+b[0])/2, (a[1]+b[1])/2)

def compute_pose_tokens(
    persons: List[Dict],    # {"keypoints":[(x,y,conf),...], "box":[...], "conf":float}
    screen_boxes: List[List[float]] = None
) -> Tuple[List[str], Dict[str, List[str]]]:
    tokens: Set[str] = set(); trace: Dict[str, List[str]] = {}
    def add(tok, why): tokens.add(tok); trace.setdefault(tok, []).append(why)
    screen_boxes = screen_boxes or []

    for p in persons:
        kps = p.get("keypoints") or []
        if len(kps) < 17: 
            continue
        sL, sR = kps[KP["l_shoulder"]], kps[KP["r_shoulder"]]
        hL, hR = kps[KP["l_hip"]], kps[KP["r_hip"]]
        shoulder_mid = _mid(sL, sR); hip_mid = _mid(hL, hR)
        spine = _vec(hip_mid, shoulder_mid)
        vertical = (0, -1)

        trunk_angle = _ang(spine, vertical)
        if trunk_angle > 45:
            add("trunk_flexion_high", f"tronco {trunk_angle:.1f}°")
        elif trunk_angle > 20:
            add("trunk_flexion_mid", f"tronco {trunk_angle:.1f}°")

        neck_vec = _vec(shoulder_mid, kps[0])
        neck_angle = _ang(neck_vec, vertical)
        if neck_angle > 30:
            add("neck_flexion_high", f"cuello {neck_angle:.1f}°")

        # pantalla por debajo de los ojos
        for sb in (screen_boxes or []):
            sx1, sy1, sx2, sy2 = sb
            screen_center_y = (sy1 + sy2)/2
            eyes_y = kps[0][1]
            if screen_center_y > eyes_y + 10:
                add("screen_below_eyes", "pantalla por debajo de la línea de ojos")
                break

    return sorted(tokens), trace
