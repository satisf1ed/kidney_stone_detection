import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from owlvit._common import load_prompts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--prompts_yaml", default="configs/prompts.yaml")
    ap.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", type=int, default=15, help="threshold grid in [0.05..0.8]")
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    meta = json.loads((coco_root/"meta.json").read_text(encoding="utf-8"))
    images_dir = Path(meta["val_images_dir"])
    ann_path = coco_root/"annotations_val.json"
    coco_gt = COCO(str(ann_path))
    img_ids = coco_gt.getImgIds()

    prompts = load_prompts(Path(args.prompts_yaml))
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.checkpoint).to(device).eval()

    raw_preds = []
    for img_id in tqdm(img_ids, desc="predict(val)"):
        info = coco_gt.loadImgs([img_id])[0]
        image = Image.open(images_dir/info["file_name"]).convert("RGB")
        inputs = processor(text=prompts, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[info["height"], info["width"]]], device=device)
        res = processor.post_process_object_detection(outputs, threshold=0.001, target_sizes=target_sizes)[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        for box, score in zip(boxes, scores):
            x1,y1,x2,y2 = box.tolist()
            raw_preds.append({"image_id": int(img_id), "category_id": 1,
                              "bbox": [x1,y1,x2-x1,y2-y1], "score": float(score)})

    thresholds = np.linspace(0.05, 0.8, args.grid).tolist()
    best = None
    out_dir = Path("runs/owlvit"); out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir/"_tmp_pred.json"

    for thr in thresholds:
        preds = [p for p in raw_preds if p["score"] >= float(thr)]
        tmp.write_text(json.dumps(preds), encoding="utf-8")
        coco_dt = coco_gt.loadRes(str(tmp))
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate(); ev.accumulate()
        map50 = float(ev.stats[1])
        if best is None or map50 > best["mAP50"]:
            best = {"best_threshold": float(thr), "mAP50": map50, "mAP50_95": float(ev.stats[0]),
                    "checkpoint": args.checkpoint}

    (out_dir/"calibration.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Best:", best)
    print("Saved:", out_dir/"calibration.json")

if __name__ == "__main__":
    main()
