"""
Genera un dataset sintético en formato YOLO para SST-RiskGPT Vision.
Crea imágenes y sus .txt "espejo" (labels) que activan los 3 riesgos:
- A: Atropellamiento/Atrapamiento (person + {forklift|truck|car|excavator|bus})
- B: Contacto con partes móviles (person + {conveyor|machine|saw|press})
- C: Caídas al mismo nivel (person + {pallet|cable|spill|toolbox})

Uso:
  python gen_synthetic_sst.py --root D:/datasets/sst --train 600 --val 200 --test 100 --imgw 1280 --imgh 720
"""

import argparse
import random
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image, ImageDraw
import math
import json

CLASS_ORDER = [
    "person", "forklift", "truck", "car", "excavator", "bus",
    "conveyor", "machine", "saw", "press",
    "pallet", "cable", "spill", "toolbox"
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}

def clamp(v, lo, hi): return max(lo, min(hi, v))

def rand_color():
    return tuple(random.randint(40, 200) for _ in range(3))

def rand_bg(w, h):
    # fondo simple con rectángulos aleatorios
    img = Image.new("RGB", (w, h), rand_color())
    d = ImageDraw.Draw(img)
    for _ in range(10):
        x1 = random.randint(0, w-1)
        x2 = clamp(x1 + random.randint(30, 200), 0, w-1)
        y1 = random.randint(0, h-1)
        y2 = clamp(y1 + random.randint(30, 200), 0, h-1)
        d.rectangle([x1, y1, x2, y2], outline=rand_color(), width=random.randint(2, 6))
    return img

def box_from_xywh(x, y, w, h, W, H) -> Tuple[int, int, int, int]:
    x1 = clamp(int(x), 0, W-1); y1 = clamp(int(y), 0, H-1)
    x2 = clamp(int(x+w), 0, W-1); y2 = clamp(int(y+h), 0, H-1)
    if x2 <= x1: x2 = clamp(x1+1, 0, W-1)
    if y2 <= y1: y2 = clamp(y1+1, 0, H-1)
    return x1, y1, x2, y2

def yolo_from_xyxy(x1, y1, x2, y2, W, H) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh

def draw_rect(d: ImageDraw.ImageDraw, x1, y1, x2, y2, color, label=None):
    d.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if label:
        d.text((x1+3, y1+3), label, fill=color)


def add_person(draw, W, H) -> Tuple[Tuple[int,int,int,int], str]:
    w = random.randint(int(0.04*W), int(0.08*W))
    h = random.randint(int(0.18*H), int(0.28*H))
    x = random.randint(0, W-w-1)
    y = random.randint(int(0.55*H), int(0.75*H))
    box = box_from_xywh(x, y, w, h, W, H)
    draw_rect(draw, *box, (240, 80, 80), "person")
    return box, "person"

def add_vehicle(draw, W, H) -> Tuple[Tuple[int,int,int,int], str]:
    kind = random.choice(["forklift","truck","car","excavator","bus"])
    w = random.randint(int(0.10*W), int(0.25*W))
    h = random.randint(int(0.10*H), int(0.22*H))
    x = random.randint(0, W-w-1)
    y = random.randint(int(0.55*H), int(0.80*H))
    box = box_from_xywh(x, y, w, h, W, H)
    draw_rect(draw, *box, (80, 160, 240), kind)
    return box, kind

def add_machine_like(draw, W, H) -> Tuple[Tuple[int,int,int,int], str]:
    kind = random.choice(["conveyor", "machine", "saw", "press"])
    w = random.randint(int(0.15*W), int(0.35*W))
    h = random.randint(int(0.12*H), int(0.22*H))
    x = random.randint(0, W-w-1)
    y = random.randint(int(0.45*H), int(0.75*H))
    box = box_from_xywh(x, y, w, h, W, H)
    draw_rect(draw, *box, (120, 220, 120), kind)
    return box, kind

def add_obstacle(draw, W, H) -> Tuple[Tuple[int,int,int,int], str]:
    kind = random.choice(["pallet", "cable", "spill", "toolbox"])
    if kind in ("pallet", "toolbox"):
        w = random.randint(int(0.10*W), int(0.20*W)) if kind=="pallet" else random.randint(int(0.06*W), int(0.12*W))
        h = random.randint(int(0.05*H), int(0.12*H)) if kind=="pallet" else random.randint(int(0.05*H), int(0.10*H))
        x = random.randint(0, W-w-1)
        y = random.randint(int(0.65*H), int(0.88*H))
        box = box_from_xywh(x, y, w, h, W, H)
        color = (240, 200, 80) if kind=="pallet" else (200, 140, 40)
        draw_rect(draw, *box, color, kind)
        return box, kind
    elif kind == "cable":
        x1 = random.randint(int(0.05*W), int(0.85*W))
        y1 = random.randint(int(0.60*H), int(0.90*H))
        angle = random.uniform(-0.6, 0.6)
        length = random.randint(int(0.25*W), int(0.60*W))
        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        x1 = clamp(x1, 0, W-1); x2 = clamp(x2, 0, W-1); y1 = clamp(y1, 0, H-1); y2 = clamp(y2, 0, H-1)
        draw.line([x1, y1, x2, y2], fill=(200, 200, 200), width=random.randint(6, 12))
        xlo, xhi = min(x1, x2), max(x1, x2)
        ylo, yhi = min(y1, y2), max(y1, y2)
        box = (xlo, ylo, xhi, yhi)
        draw_rect(draw, *box, (200, 200, 200), "cable")
        return box, kind
    else:  # spill
        w = random.randint(int(0.10*W), int(0.20*W))
        h = random.randint(int(0.05*H), int(0.12*H))
        x = random.randint(0, W-w-1)
        y = random.randint(int(0.70*H), int(0.90*H))
        box = box_from_xywh(x, y, w, h, W, H)
        draw.ellipse(box, outline=(80, 80, 255), width=3)
        draw_rect(draw, *box, (80, 80, 255), "spill")
        return box, kind

def near(b1, b2, max_dist=160) -> bool:
    c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
    c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    return math.dist(c1, c2) <= max_dist

def place_pair_near(draw, W, H, fnA, fnB):
    boxA, a = fnA(draw, W, H)
    for _ in range(20):
        boxB, b = fnB(draw, W, H)
        if near(boxA, boxB, max_dist=random.randint(80, 220)):
            return (boxA, a), (boxB, b)
    return (boxA, a), (boxB, b)

def normalize_label(item):
    """Acepta (clase, box) o (box, clase) y devuelve (clase, box)."""
    if not isinstance(item, tuple) or len(item) != 2:
        raise ValueError(f"Etiqueta inesperada: {item}")
    a, b = item
    if isinstance(a, str) and isinstance(b, tuple) and len(b) == 4:
        return a, b
    if isinstance(b, str) and isinstance(a, tuple) and len(a) == 4:
        return b, a
    raise ValueError(f"Etiqueta inesperada: {item}")

def save_label(path_txt: Path, labels: List[Tuple[str, Tuple[int,int,int,int]]], W, H):
    lines = []
    for it in labels:
        cname, (x1, y1, x2, y2) = normalize_label(it)
        cx, cy, bw, bh = yolo_from_xyxy(x1, y1, x2, y2, W, H)
        cid = CLASS_TO_ID[cname]
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path_txt.write_text("\n".join(lines), encoding="utf-8")

def generate_split(root: Path, split: str, n_images: int, W: int, H: int, seed: int):
    random.seed(seed)
    (root / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (root / f"labels/{split}").mkdir(parents=True, exist_ok=True)
    stats = {"total": n_images, "scenario_counts": {"A":0,"B":0,"C":0,"NEG":0}}

    for i in range(n_images):
        img = rand_bg(W, H)
        d = ImageDraw.Draw(img)
        labels: List[Tuple[str, Tuple[int,int,int,int]]] = []

        scenario = random.choices(["A","B","C","NEG"], weights=[0.35, 0.28, 0.28, 0.09])[0]

        if scenario == "A":
            (b1, a1), (b2, a2) = place_pair_near(d, W, H, add_person, add_vehicle)
            labels += [(a1, b1), (a2, b2)]
            if random.random() < 0.3:
                b3, a3 = add_obstacle(d, W, H)
                labels.append((a3, b3))

        elif scenario == "B":
            (b1, a1), (b2, a2) = place_pair_near(d, W, H, add_person, add_machine_like)
            labels += [(a1, b1), (a2, b2)]
            if random.random() < 0.3:
                b3, a3 = add_obstacle(d, W, H)
                labels.append((a3, b3))

        elif scenario == "C":
            (b1, a1), (b2, a2) = place_pair_near(d, W, H, add_person, add_obstacle)
            labels += [(a1, b1), (a2, b2)]
            if random.random() < 0.25:
                b3, a3 = add_obstacle(d, W, H)
                labels.append((a3, b3))

        else: 
            if random.random() < 0.5:
                b, a = add_person(d, W, H);      labels.append((a, b))
            if random.random() < 0.5:
                b, a = add_vehicle(d, W, H);     labels.append((a, b))
            if random.random() < 0.5:
                b, a = add_machine_like(d, W, H);labels.append((a, b))
            if random.random() < 0.5:
                b, a = add_obstacle(d, W, H);    labels.append((a, b))

        img_name = f"{split}_{i:05d}.jpg"
        lab_name = f"{split}_{i:05d}.txt"
        img.save(root / f"images/{split}/{img_name}", quality=92)
        save_label(root / f"labels/{split}/{lab_name}", labels, W, H)
        stats["scenario_counts"][scenario] += 1

    (root / f"stats_{split}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[OK] Generado split {split}: {n_images} imágenes → {root}")

def write_dataset_yaml(root: Path):
    content = "path: " + str(root).replace("\\","/") + "\n"
    content += "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
    for i, c in enumerate(CLASS_ORDER):
        content += f"  {i}: {c}\n"
    (root / "dataset.yaml").write_text(content, encoding="utf-8")
    print(f"[OK] Escrito dataset.yaml en {root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sst", help="Raíz del dataset sintético")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val",   type=int, default=200)
    ap.add_argument("--test",  type=int, default=100)
    ap.add_argument("--imgw", type=int, default=1280)
    ap.add_argument("--imgh", type=int, default=720)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    root = Path(args.root)
    print(f"Generando dataset sintético en: {root}")
    write_dataset_yaml(root)
    generate_split(root, "train", args.train, args.imgw, args.imgh, args.seed+1)
    generate_split(root, "val",   args.val,   args.imgw, args.imgh, args.seed+2)
    generate_split(root, "test",  args.test,  args.imgw, args.imgh, args.seed+3)
    print("\nSugerencia: revisa stats_train.json / stats_val.json para ver distribución de escenarios.")

if __name__ == "__main__":
    main()
