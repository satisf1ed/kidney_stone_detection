import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import matplotlib.pyplot as plt

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


def roc_curve_from_scores(y_true: np.ndarray, y_score: np.ndarray):
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())

    uniq_scores, idx = np.unique(y_score, return_index=True)
    idx = idx[np.argsort(-uniq_scores)]
    thresholds = y_score[idx]

    tp_cum = np.cumsum(y_true == 1)
    fp_cum = np.cumsum(y_true == 0)

    tpr = []
    fpr = []
    for i in idx:
        tp = tp_cum[i]
        fp = fp_cum[i]
        tpr.append(tp / P)
        fpr.append(fp / N)

    tpr = np.array([0.0] + tpr, dtype=np.float64)
    fpr = np.array([0.0] + fpr, dtype=np.float64)
    thresholds = np.array([thresholds[0] + 1e-9] + thresholds.tolist(), dtype=np.float64)

    tpr = np.append(tpr, 1.0)
    fpr = np.append(fpr, 1.0)
    thresholds = np.append(thresholds, thresholds[-1] - 1e-9)

    return fpr, tpr, thresholds


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapz(y, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    ap.add_argument("--weights", required=True, help="Path to YOLO best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max_images", type=int, default=0, help="0=all, else limit")
    ap.add_argument("--out", default="runs/bench/roc_yolo.png")
    ap.add_argument("--save_scores", action="store_true", help="Save y_true/y_score npz next to plot")

    ap.add_argument("--human_sens", type=float, default=0.57, help="Human sensitivity (TPR)")
    ap.add_argument("--human_spec", type=float, default=0.73, help="Human specificity (TNR)")
    ap.add_argument("--plot_human", action="store_true", help="Overlay 3-point 'human ROC' and compute its AUC")

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
    y_score = []

    for im in tqdm(images, desc=f"predict scores ({args.split})"):
        gt_pos = 1 if image_has_gt(labels_dir, images_dir, im) else 0
        y_true.append(gt_pos)

        r = model.predict(
            source=str(im),
            conf=0.001,
            iou=0.7,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            y_score.append(0.0)
        else:
            y_score.append(float(r.boxes.conf.max().cpu().item()))

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float64)

    fpr, tpr, thr = roc_curve_from_scores(y_true, y_score)
    roc_auc = auc_trapz(fpr, tpr)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    print("\nROC")
    print(f"split={args.split} images={len(images)} positives={n_pos} negatives={n_neg}")
    print(f"YOLO AUC={roc_auc:.6f}")

    human_auc = None
    human_fpr = None
    human_tpr = None

    if args.plot_human:
        human_tpr = float(args.human_sens)
        human_fpr = float(1.0 - args.human_spec)
        hx = np.array([0.0, human_fpr, 1.0], dtype=np.float64)
        hy = np.array([0.0, human_tpr, 1.0], dtype=np.float64)
        human_auc = auc_trapz(hx, hy)
        print("\n'Human ROC' 3 points")
        print(f"Human point: TPR(sens)={human_tpr:.4f}, FPR(1-spec)={human_fpr:.4f}")
        print(f"Human AUC (3-point)={human_auc:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"YOLO ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")

    if args.plot_human:
        hx = [0.0, human_fpr, 1.0]
        hy = [0.0, human_tpr, 1.0]
        plt.plot(hx, hy, label=f"Human 3-point ROC (AUC={human_auc:.4f})")
        plt.scatter([human_fpr], [human_tpr], label="Human operating point")

    plt.xlabel("False Positive Rate (1 - specificity)")
    plt.ylabel("True Positive Rate (sensitivity)")
    title = f"ROC — split={args.split}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved ROC plot: {out_path}")

    if args.save_scores:
        npz_path = out_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            y_true=y_true,
            y_score=y_score,
            fpr=fpr,
            tpr=tpr,
            thresholds=thr,
            yolo_auc=roc_auc,
            human_sens=args.human_sens,
            human_spec=args.human_spec,
            human_auc_3pt=human_auc if human_auc is not None else np.nan,
        )
        print(f"Saved scores: {npz_path}")


if __name__ == "__main__":
    main()
