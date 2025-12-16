import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def load_split_dirs(data_yaml: Path, split: str):
    import yaml

    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent

    key = split
    if split == "val" and d.get("val") is None and d.get("valid") is not None:
        key = "valid"

    split_path = resolve_path(d[key], base)

    images_dir = split_path
    if images_dir.name != "images" and (images_dir / "images").exists():
        images_dir = images_dir / "images"

    labels_dir = images_dir.parent / "labels"
    if not labels_dir.exists():
        candidate = images_dir.parent.parent / "labels"
        if candidate.exists():
            labels_dir = candidate

    return images_dir, labels_dir


def image_has_gt(labels_dir: Path, images_dir: Path, img_path: Path) -> bool:
    lab = (labels_dir / img_path.relative_to(images_dir)).with_suffix(".txt")
    if not lab.exists():
        return False
    return bool(lab.read_text().strip())


def compute_sens_spec(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "accuracy": acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    ap.add_argument("--weights", required=True, help="Path to YOLO best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.25, help="Prediction conf threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    ap.add_argument("--max_images", type=int, default=0, help="0=all, else limit for quick test")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    images_dir, labels_dir = load_split_dirs(data_yaml, args.split)

    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    images.sort()
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    model = YOLO(args.weights)

    y_true = []
    y_pred = []

    for im in tqdm(images, desc=f"predict ({args.split})"):
        gt_pos = 1 if image_has_gt(labels_dir, images_dir, im) else 0
        y_true.append(gt_pos)

        r = model.predict(
            source=str(im),
            conf=args.conf,
            iou=0.7,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )[0]

        pred_pos = 1 if (r.boxes is not None and len(r.boxes) > 0) else 0
        y_pred.append(pred_pos)

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)

    m = compute_sens_spec(y_true, y_pred)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    print(f"split={args.split} images={len(images)} positives={n_pos} negatives={n_neg}")
    print(f"conf={args.conf}")
    for k in ["TP", "FP", "TN", "FN", "sensitivity", "specificity", "precision", "accuracy"]:
        v = m[k]
        if isinstance(v, float):
            print(f"{k:>12}: {v:.4f}")
        else:
            print(f"{k:>12}: {v}")

    out_dir = Path("runs/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sens_spec_{args.split}.txt").write_text(
        "\n".join([f"{k}: {m[k]}" for k in m]) + "\n", encoding="utf-8"
    )
    print(f"\nSaved: {out_dir / f'sens_spec_{args.split}.txt'}")


if __name__ == "__main__":
    main()
