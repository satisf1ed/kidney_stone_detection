import argparse, random
from pathlib import Path
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def resolve(path, base):
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p

def images_from_data_yaml(data_yaml: Path, split: str):
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    key = split
    if split == "val" and d.get("val") is None and d.get("valid") is not None:
        key = "valid"
    sp = resolve(d.get(key), base)
    if sp.name != "images" and (sp / "images").exists():
        sp = sp / "images"
    imgs = [p for p in sp.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="runs/yolo11/infer")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    imgs = images_from_data_yaml(data_yaml, args.split)
    random.shuffle(imgs)
    imgs = imgs[:args.n]

    model = YOLO(args.weights)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for im in imgs:
        model.predict(source=str(im), conf=args.conf, device=args.device,
                      save=True, project=args.out, name=".", exist_ok=True, verbose=False)
    print("Saved to:", args.out)

if __name__ == "__main__":
    main()
