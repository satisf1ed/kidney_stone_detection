import argparse
from pathlib import Path
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def resolve(path, base):
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p

def normalize_images_dir(p: Path):
    if p.name == "images" and p.exists():
        return p
    if (p / "images").exists():
        return p / "images"
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path or glob to data.yaml")
    args = ap.parse_args()

    paths = list(Path(".").glob(args.data)) if any(ch in args.data for ch in ["*", "?", "["]) else [Path(args.data)]

    for data_yaml in paths:
        d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        base = Path(data_yaml).parent

        def get_split(key):
            v = d.get(key)
            if v is None:
                return None
            return normalize_images_dir(resolve(v, base))

        train = get_split("train")
        val = get_split("val") or get_split("valid")
        test = get_split("test")

        print(f"\n{data_yaml}")
        for name, img_dir in [("train", train), ("val", val), ("test", test)]:
            if img_dir is None:
                print(f"{name:>5}: (missing)")
                continue
            if not img_dir.exists():
                print(f"{name:>5}: images_dir not found -> {img_dir}")
                continue

            lbl_dir = img_dir.parent / "labels"
            imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
            missing, empty = 0, 0
            for im in imgs:
                rel = im.relative_to(img_dir)
                lab = (lbl_dir / rel).with_suffix(".txt")
                if not lab.exists():
                    missing += 1
                else:
                    if lab.read_text().strip() == "":
                        empty += 1
            print(f"{name:>5}: images={len(imgs)} labels_missing={missing} empty_labels={empty} dir={img_dir}")

if __name__ == "__main__":
    main()
