import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="pesos base (n/s/m/l/x)")
    ap.add_argument("--data", required=True, help="dataset.yaml (YOLO) o COCO convertido")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=-1)  # auto
    ap.add_argument("--device", default=None, help="'0' para GPU")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="sst_yolo")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )

if __name__ == "__main__":
    main()
