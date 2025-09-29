from ultralytics import YOLO
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ruta a data.yaml (formato Ultralytics)")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        lr0=args.lr0,
        mosaic=1.0, mixup=0.2, copy_paste=0.4,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.10, scale=0.6, shear=0.0, perspective=0.0005,
        degrees=5.0,
        fliplr=0.5, flipud=0.1,
        val=True,
        device=0
    )

if __name__ == "__main__":
    main()
