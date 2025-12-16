import argparse, random, json
from pathlib import Path
import yaml
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def resolve(path, base):
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p

def get_images(data_yaml: Path, split: str):
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    key = split
    if split == "val" and d.get("val") is None and d.get("valid") is not None:
        key = "valid"
    sp = resolve(d.get(key), base)
    if sp.name != "images" and (sp/"images").exists():
        sp = sp/"images"
    imgs = [p for p in sp.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="facebook/detr-resnet-50")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/detr/infer")
    args = ap.parse_args()

    imgs = get_images(Path(args.data_yaml), args.split)
    random.shuffle(imgs)
    imgs = imgs[:args.n]

    device = torch.device(args.device)
    processor = DetrImageProcessor.from_pretrained(args.model)
    model = DetrForObjectDetection.from_pretrained(
        args.model,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for im in imgs:
        image = Image.open(im).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        w,h = image.size
        target_sizes = torch.tensor([[h,w]], device=device)
        res = processor.post_process_object_detection(outputs, threshold=0.001, target_sizes=target_sizes)[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for box, score in zip(boxes, scores):
            if float(score) < args.conf: 
                continue
            x1,y1,x2,y2 = map(int, box.tolist())
            cv2.rectangle(img_bgr, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img_bgr, f"{score:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(str(out_dir/im.name), img_bgr)

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
