"""
Evalúa confiabilidad (objetivo 2.6):
- Ejecuta detección + inferencia de riesgos sobre cada imagen del CSV
- Compara con etiquetas de expertos (riesgos ';' separados)
- Reporta Precision/Recall/F1 (micro) y Kappa de Cohen
"""
import argparse, csv
from typing import List, Set
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from detector import YoloDetector
from rules_engine import RiskEngine
from PIL import Image

def tokenize(s: str) -> Set[str]:
    return set(x.strip() for x in s.split(";") if x.strip())

def binarize(y_true, y_pred, labels):
    Yt, Yp = [], []
    for t, p in zip(y_true, y_pred):
        Yt.extend([1 if lab in t else 0 for lab in labels])
        Yp.extend([1 if lab in p else 0 for lab in labels])
    return Yt, Yp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con columnas: image_path,risks")
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()

    det = YoloDetector(model_path=args.model)
    eng = RiskEngine("risk_ontology.yaml")

    gt_sets, pr_sets = [], []
    labels_universe = set()

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = Image.open(row["image_path"]).convert("RGB")
            dets = det.to_dicts(det.predict(img))
            present = det.classes_from_dicts(dets)
            risks = eng.infer(present)
            pred_ids = set(r["id"] for r in risks)

            true_ids = tokenize(row["risks"])

            gt_sets.append(true_ids)
            pr_sets.append(pred_ids)
            labels_universe |= true_ids | pred_ids

    labels = sorted(labels_universe) or ["atrapamiento_atropellamiento","contacto_partes_moviles","caidas_mismo_nivel_obstaculos"]
    y_true_bin, y_pred_bin = binarize(gt_sets, pr_sets, labels)

    P, R, F1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    K = cohen_kappa_score(y_true_bin, y_pred_bin)

    print({"labels": labels, "precision_micro": P, "recall_micro": R, "f1_micro": F1, "cohen_kappa": K})

if __name__ == "__main__":
    main()
