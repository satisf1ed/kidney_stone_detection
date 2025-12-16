import argparse, json
from pathlib import Path
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
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--prompts_yaml", default="configs/prompts.yaml")
    ap.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score_threshold", type=float, default=None)
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    meta = json.loads((coco_root/"meta.json").read_text(encoding="utf-8"))
    images_dir = Path(meta[f"{args.split}_images_dir"])
    ann_path = coco_root/f"annotations_{args.split}.json"
    coco_gt = COCO(str(ann_path))
    img_ids = coco_gt.getImgIds()

    thr = args.score_threshold
    if thr is None:
        calib = Path("runs/owlvit/calibration.json")
        thr = json.loads(calib.read_text()).get("best_threshold", 0.25) if calib.exists() else 0.25

    prompts = load_prompts(Path(args.prompts_yaml))
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.checkpoint).to(device).eval()

    preds = []
    for img_id in tqdm(img_ids, desc="predict"):
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
            if float(score) < float(thr):
                continue
            x1,y1,x2,y2 = box.tolist()
            preds.append({"image_id": int(img_id), "category_id": 1,
                          "bbox": [x1,y1,x2-x1,y2-y1], "score": float(score)})

    out_dir = Path("runs/bench/owlvit"); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir/f"pred_{args.split}.json"
    pred_path.write_text(json.dumps(preds), encoding="utf-8")

    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    metrics = {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]),
               "threshold": float(thr), "checkpoint": args.checkpoint}
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", out_dir/"metrics.json")

if __name__ == "__main__":
    main()
