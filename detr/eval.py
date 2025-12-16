import argparse, json
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="facebook/detr-resnet-50")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    meta = json.loads((coco_root/"meta.json").read_text(encoding="utf-8"))
    images_dir = Path(meta[f"{args.split}_images_dir"])
    ann_path = coco_root/f"annotations_{args.split}.json"
    coco_gt = COCO(str(ann_path))
    ids = coco_gt.getImgIds()

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

    preds = []
    for img_id in tqdm(ids, desc="predict"):
        info = coco_gt.loadImgs([img_id])[0]
        img = Image.open(images_dir/info["file_name"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[info["height"], info["width"]]], device=device)
        res = processor.post_process_object_detection(outputs, threshold=args.conf, target_sizes=target_sizes)[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        for box,score,lab in zip(boxes,scores,labels):
            x1,y1,x2,y2 = box.tolist()
            preds.append({
                "image_id": int(img_id),
                "category_id": 1,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "score": float(score),
            })

    out_dir = Path("runs/bench/detr"); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir/f"pred_{args.split}.json"
    pred_path.write_text(json.dumps(preds), encoding="utf-8")

    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    metrics = {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1])}
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", out_dir/"metrics.json")

if __name__ == "__main__":
    main()
