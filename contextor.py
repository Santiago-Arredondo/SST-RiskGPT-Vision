from __future__ import annotations
from typing import List, Dict, Any, Set, Tuple
import numpy as np
import cv2

def _clip(a, lo, hi): 
    return int(max(lo, min(hi, a)))

def _line_len(px1, py1, px2, py2):
    dx, dy = px2 - px1, py2 - py1
    return (dx*dx + dy*dy) ** 0.5

def _hough_lines(img_edges, min_len=40, max_gap=10):
    lines = cv2.HoughLinesP(img_edges, 1, np.pi/180, threshold=70,
                            minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []
    return [l[0] for l in lines]  # (x1,y1,x2,y2)

def _ratio_vertical_horizontal(lines):
    v = h = d = 0
    for x1,y1,x2,y2 in lines:
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dx + dy == 0:
            continue
        if dx <= 5 and dy >= 10:
            v += 1
        elif dy <= 5 and dx >= 10:
            h += 1
        else:
            d += 1
    return v, h, d

def _roi(img, box, pad=0.15):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 = _clip(x1 - int(pad*w), 0, W-1)
    y1 = _clip(y1 - int(pad*h), 0, H-1)
    x2 = _clip(x2 + int(pad*w), 1, W)
    y2 = _clip(y2 + int(pad*h), 1, H)
    return img[y1:y2, x1:x2]

def compute_context_tokens(
    frame_bgr: np.ndarray,
    dets: List[Dict[str, Any]],
    *,
    strong_mode: bool = False
) -> Set[str]:
    """
    Devuelve tokens heurísticos: {"construction_context","ladder_like","edge_exposure"}
    """
    toks: Set[str] = set()
    H, W = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 80, 160)

    # --- Escaneo global para "contexto de construcción" (muchas líneas verticales) ---
    lines = _hough_lines(edges, min_len=40, max_gap=10)
    v_cnt, h_cnt, _ = _ratio_vertical_horizontal(lines)
    # densidad por megapíxel
    mpix = max(1.0, (W*H)/1_000_000.0)
    if v_cnt/mpix > (35 if strong_mode else 22):
        toks.add("construction_context")

    # --- Necesitamos persona para heurísticas focalizadas ---
    persons = [d for d in dets if d.get("cls") == "person"]
    if not persons:
        return toks

    # Tomamos la persona más grande (normalmente la principal)
    p = max(persons, key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]))
    x1,y1,x2,y2 = map(int, p["box"])
    roi = _roi(frame_bgr, (x1,y1,x2,y2), pad=0.25)
    if roi.size > 0:
        roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_g = cv2.GaussianBlur(roi_g, (3,3), 0)
        roi_edges = cv2.Canny(roi_g, 80, 160)
        roi_lines = _hough_lines(roi_edges, min_len=25, max_gap=6)
        rv, rh, _ = _ratio_vertical_horizontal(roi_lines)

        # "ladder_like": en ROI hay combo de varias líneas verticales y horizontales (paralelas)
        if rv >= (5 if strong_mode else 3) and rh >= (5 if strong_mode else 3):
            toks.add("ladder_like")

    # --- "edge_exposure": persona alta y existe línea larga horizontal tipo losa justo por debajo ---
    feet_y = y2
    # banda horizontal justo por debajo de los pies (si existe)
    band_top = _clip(feet_y + int(0.01*H), 0, H-1)
    band_bot = _clip(feet_y + int(0.10*H), 0, H-1)
    if band_bot > band_top:
        band = edges[band_top:band_bot, :]
        blines = _hough_lines(band, min_len=int(0.35*W), max_gap=15)
        # ¿alguna línea casi horizontal y larga?
        long_h = 0
        for x1b,y1b,x2b,y2b in blines:
            if abs(y2b - y1b) <= 4 and (x2b - x1b) >= int(0.35*W):
                long_h += 1
        # persona ubicada en mitad superior -> "posible borde"
        if long_h >= 1 and feet_y < int(0.65*H):
            toks.add("edge_exposure")

    return toks
