import argparse, json, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device)

    out = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    os.makedirs("runs/bench/yolo11", exist_ok=True)
    with open("runs/bench/yolo11/metrics.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved runs/bench/yolo11/metrics.json")

if __name__ == "__main__":
    main()
