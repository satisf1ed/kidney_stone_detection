import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path/glob to data.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--project", default="runs/yolo11")
    ap.add_argument("--name", default="train")
    ap.add_argument("--mosaic", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        project=args.project,
        name=args.name,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=0,
    )
    print("Best:", Path(args.project)/args.name/"weights"/"best.pt")

if __name__ == "__main__":
    main()
