import argparse
import json
from pathlib import Path
from PIL import Image
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def resolve(path, base):
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p

def normalize_images_dir(p: Path):
    if p is None:
        return None
    if p.name == "images" and p.exists():
        return p
    if (p / "images").exists():
        return p / "images"
    return p

def parse_names(names):
    if names is None:
        return ["kidney-stone"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return names
    return [str(names)]

def build_split(images_dir: Path, out_json: Path, names):
    if images_dir is None or not images_dir.exists():
        return
    labels_dir = images_dir.parent / "labels"
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    images.sort()

    names_list = parse_names(names)
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names_list)]
    cat_id_map = {i: i + 1 for i in range(len(names_list))}

    coco = {"images": [], "annotations": [], "categories": categories}

    ann_id = 1
    for img_id, img_path in enumerate(images, start=1):
        w, h = Image.open(img_path).size
        rel = img_path.relative_to(images_dir).as_posix()
        coco["images"].append({"id": img_id, "file_name": rel, "width": w, "height": h})

        lab = (labels_dir / rel).with_suffix(".txt")
        if lab.exists():
            txt = lab.read_text().strip()
            if txt:
                for line in txt.splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x = (cx - bw / 2.0) * w
                    y = (cy - bh / 2.0) * h
                    ww = bw * w
                    hh = bh * h
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id_map.get(cls, cls + 1),
                        "bbox": [x, y, ww, hh],
                        "area": float(max(0.0, ww) * max(0.0, hh)),
                        "iscrowd": 0
                    })
                    ann_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print("Wrote:", out_json, "images:", len(coco["images"]), "anns:", len(coco["annotations"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--out", required=True, help="Output folder for COCO jsons")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent

    train = normalize_images_dir(resolve(d.get("train"), base)) if d.get("train") else None
    val = normalize_images_dir(resolve(d.get("val") or d.get("valid"), base)) if (d.get("val") or d.get("valid")) else None
    test = normalize_images_dir(resolve(d.get("test"), base)) if d.get("test") else None
    names = d.get("names")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"train_images_dir": str(train) if train else None,
            "val_images_dir": str(val) if val else None,
            "test_images_dir": str(test) if test else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    build_split(train, out / "annotations_train.json", names)
    build_split(val, out / "annotations_val.json", names)
    build_split(test, out / "annotations_test.json", names)

if __name__ == "__main__":
    main()
