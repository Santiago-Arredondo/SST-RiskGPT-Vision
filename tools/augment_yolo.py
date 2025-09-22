# tools/augment_yolo.py
"""
Aumenta datasets YOLO (imágenes + labels) de forma segura con Albumentations.
- Lee y escribe bboxes en formato YOLO (x_center, y_center, w, h) normalizado.
- Aplica transformaciones comunes sin romper cajas.
- Evita APIs internas/deprecadas de Albumentations.

Uso típico:
  python tools/augment_yolo.py --root "D:/datasets/sst" --split train --mult 2 --seed 1337

Estructura esperada del dataset:
  <root>/
    images/
      train/  val/  test/
    labels/
      train/  val/  test/
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import random
import cv2
import numpy as np

try:
    import albumentations as A
except Exception as e:
    raise SystemExit(
        "Albumentations no está instalado. Instálalo así:\n"
        "  pip install 'albumentations>=1.3,<2.0'  opencv-python"
    ) from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_yolo_labels(lbl_path: Path) -> tuple[list[int], list[list[float]]]:
    """Lee un .txt YOLO: 'cls x y w h' (normalizado). Devuelve (class_ids, bboxes)."""
    ids: list[int] = []
    boxes: list[list[float]] = []
    if not lbl_path.exists():
        return ids, boxes
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            ids.append(cls)
            boxes.append([x, y, w, h])
    return ids, boxes


def write_yolo_labels(lbl_path: Path, class_ids: list[int], boxes: list[list[float]]) -> None:
    """Escribe un .txt YOLO 'cls x y w h' (normalizado)."""
    if not class_ids or not boxes:
        # Si no hay cajas, deja el archivo vacío (o bórralo si prefieres).
        lbl_path.write_text("", encoding="utf-8")
        return

    lines = []
    for cls, (x, y, w, h) in zip(class_ids, boxes):
        lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_transform() -> A.Compose:
    """
    Conjunto de transforms compatibles y estables.
    - Usamos A.Affine en vez de ShiftScaleRotate para evitar el warning.
    - GaussNoise con var_limit válido (en algunas versiones se llama igual).
    """
    aug = A.Compose(
        [
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.25,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-7, 7),
                shear=(-5, 5),
                translate_percent=(0.0, 0.07),
                p=0.45,
            ),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.2
        ),
    )
    return aug


def augment_split(root: Path, split: str, mult: int, keep_empty: bool, jpeg_quality: int) -> None:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(
            f"No existe {img_dir} o {lbl_dir}. Verifica --root y --split."
        )

    tf = build_transform()
    images = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

    if not images:
        print(f"[WARN] No se encontraron imágenes en {img_dir}")
        return

    print(f"[INFO] Encontradas {len(images)} imágenes en {split}. Generando x{mult} por imagen...")

    for img_path in images:
        rel = img_path.relative_to(img_dir)
        lbl_path = lbl_dir / rel.with_suffix(".txt")
        lbl_path.parent.mkdir(parents=True, exist_ok=True)

        # Leer imagen (BGR) y convertir a RGB para Albumentations.
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] No se pudo leer: {img_path}")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        class_ids, boxes = read_yolo_labels(lbl_path)

        for k in range(mult):
            class_labels = class_ids[:]  # Albumentations necesita lista paralela a bboxes
            try:
                out = tf(image=img, bboxes=boxes, class_labels=class_labels)
                out_img = out["image"]
                out_boxes = out["bboxes"]
                out_ids = out["class_labels"]
            except Exception as e:
                print(f"[ERR] Falló aug en {img_path.name}: {e}")
                continue

            if not out_boxes and not keep_empty:
                # Si se perdieron todas las cajas y no queremos vacíos, saltamos
                continue

            # Salvar con sufijo
            stem = img_path.stem
            out_img_path = img_path.with_name(f"{stem}_aug{k+1}{img_path.suffix}")
            out_lbl_path = lbl_path.with_name(f"{stem}_aug{k+1}.txt")

            # Guardar imagen (convertimos de vuelta a BGR)
            out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            if out_img_path.suffix.lower() in {".jpg", ".jpeg"}:
                cv2.imwrite(str(out_img_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            else:
                cv2.imwrite(str(out_img_path), out_bgr)

            # Guardar labels
            write_yolo_labels(out_lbl_path, out_ids, out_boxes)

    print("[OK] Augmentación finalizada.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Raíz del dataset (ej: D:/datasets/sst)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split a aumentar")
    ap.add_argument("--mult", type=int, default=2, help="Variantes por imagen")
    ap.add_argument("--keep-empty", action="store_true", help="Guardar aunque no queden bboxes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.root)
    augment_split(root, args.split, args.mult, args.keep_empty, args.jpeg_quality)


if __name__ == "__main__":
    main()
