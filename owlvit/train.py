"""
Supervised fine-tuning for OWLv2/OWL-ViT style models (best-effort).

Почему это нужно:
- В некоторых версиях transformers Owlv2ForObjectDetection.forward() НЕ принимает labels/targets,
  поэтому стандартный подход "model(..., labels=...)" не работает.
- Этот скрипт считает supervised loss сам: Hungarian matching + BCE cls + L1 bbox + (1-GIoU).

Ограничения:
- Это "DETR-style" loss, не обязательно 1-в-1 совпадает с оригинальным обучением OWL.
- Для УЗИ обычно лучше оставить OWL как zero-shot baseline, а supervised делать YOLO/DETR.
  Но если хотите fine-tune — этот скрипт даёт рабочий цикл обучения.

Требования:
- scipy (для Hungarian matching)
"""

import argparse, json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise SystemExit(
        "scipy is required for Hungarian matching. Please `pip install scipy`.\n"
        f"Import error: {e}"
    )

# If you have your own prompt loader, keep it.
# Otherwise you can just hardcode prompts=["kidney stone"].
from owlvit._common import load_prompts


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# Box utils (normalized [0..1])
# -----------------------------
def cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area_xyxy(b: torch.Tensor) -> torch.Tensor:
    return (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor):
    # boxes1: (N,4), boxes2: (M,4)
    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter + 1e-9
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # returns (N,M) GIoU
    iou, union = box_iou_xyxy(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1] + 1e-9

    giou = iou - (area_c - union) / area_c
    return giou


def hungarian_match(cost: torch.Tensor):
    """
    cost: (Q, T) torch tensor
    returns (q_idx, t_idx) indices
    """
    if cost.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=cost.device),
            torch.empty((0,), dtype=torch.int64, device=cost.device),
        )
    c = cost.detach().cpu().numpy()
    q_idx, t_idx = linear_sum_assignment(c)
    return (
        torch.as_tensor(q_idx, dtype=torch.int64, device=cost.device),
        torch.as_tensor(t_idx, dtype=torch.int64, device=cost.device),
    )


def compute_detr_style_loss(
    logits: torch.Tensor,          # (B, Q, 1) for single-class
    pred_boxes: torch.Tensor,      # (B, Q, 4) normalized, likely cxcywh
    targets: list,                 # list of {"boxes": (T,4) xyxy normalized, "labels": (T,)}
    pred_box_format: str = "cxcywh",
    cls_weight: float = 1.0,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
):
    """
    DETR-style: match GT boxes to queries via Hungarian, then:
      - BCEWithLogits for single-class objectness (matched queries => 1, others => 0)
      - L1 on matched boxes (xyxy)
      - (1 - GIoU) on matched boxes
    """
    device = logits.device
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")

    B, Q, C = logits.shape
    assert C == 1, "This script expects single-class training (one prompt)."

    # Convert predicted boxes to xyxy
    if pred_box_format == "cxcywh":
        pred_xyxy = cxcywh_to_xyxy(pred_boxes)
    elif pred_box_format == "xyxy":
        pred_xyxy = pred_boxes
    else:
        raise ValueError(f"Unknown pred_box_format={pred_box_format}")

    total_cls = torch.tensor(0.0, device=device)
    total_l1 = torch.tensor(0.0, device=device)
    total_giou = torch.tensor(0.0, device=device)

    matched_total = 0

    for b in range(B):
        tgt = targets[b]
        tgt_xyxy = tgt["boxes"]  # (T,4) xyxy normalized
        T = tgt_xyxy.shape[0]

        # default all queries negative
        cls_tgt = torch.zeros((Q, 1), device=device, dtype=logits.dtype)

        if T == 0:
            total_cls += bce(logits[b], cls_tgt)
            continue

        # Classification cost: matched queries should have high sigmoid(logit)
        prob = torch.sigmoid(logits[b, :, 0])  # (Q,)
        cost_cls = -prob[:, None].repeat(1, T)  # (Q,T)

        # L1 cost on xyxy
        cost_l1 = torch.cdist(pred_xyxy[b], tgt_xyxy, p=1)  # (Q,T)

        # GIoU cost (maximize => minimize negative)
        giou = generalized_box_iou(pred_xyxy[b], tgt_xyxy)  # (Q,T)
        cost_giou = -giou

        cost = cls_weight * cost_cls + l1_weight * cost_l1 + giou_weight * cost_giou
        qi, ti = hungarian_match(cost)

        # mark matched queries as positive
        cls_tgt[qi, 0] = 1.0
        total_cls += bce(logits[b], cls_tgt)

        # bbox losses only for matched pairs
        pb = pred_xyxy[b][qi]      # (M,4)
        tb = tgt_xyxy[ti]          # (M,4)
        matched_total += pb.shape[0]

        total_l1 += torch.nn.functional.l1_loss(pb, tb, reduction="sum")
        # 1 - GIoU for matched boxes (diagonal)
        g = generalized_box_iou(pb, tb)
        total_giou += (1.0 - g.diag()).sum()

    # normalize by batch (or by matched count; оставим по batch чтобы стабильно)
    loss_cls = total_cls / max(1, B)
    loss_l1 = total_l1 / max(1, B)
    loss_giou = total_giou / max(1, B)

    loss = loss_cls + l1_weight * loss_l1 + giou_weight * loss_giou
    stats = {
        "loss_total": float(loss.detach().cpu()),
        "loss_cls": float(loss_cls.detach().cpu()),
        "loss_l1": float(loss_l1.detach().cpu()),
        "loss_giou": float(loss_giou.detach().cpu()),
        "matched_per_batch": float(matched_total) / max(1, B),
    }
    return loss, stats


# -----------------------------
# Dataset
# -----------------------------
class CocoForOpenVocab(Dataset):
    """
    Reads COCO JSON. Uses ONE prompt (single-class).
    Returns:
      pixel_values, input_ids, attention_mask (optional)
      targets: boxes (T,4) xyxy normalized [0..1], labels (T,) all zeros
    """
    def __init__(self, images_root: Path, ann_json: Path, processor, prompt: str):
        coco = json.loads(ann_json.read_text(encoding="utf-8"))
        self.images_root = images_root
        self.processor = processor
        self.prompt = prompt

        self.images = {im["id"]: im for im in coco["images"]}
        self.anns_by_img = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        img_path = self.images_root / info["file_name"]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        anns = self.anns_by_img.get(img_id, [])

        # GT boxes: COCO bbox = [x,y,w,h] absolute pixels
        boxes = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            x1 = x / w
            y1 = y / h
            x2 = (x + bw) / w
            y2 = (y + bh) / h
            boxes.append([x1, y1, x2, y2])

        # Single-class => all labels = 0
        labels = [0] * len(boxes)

        # Processor expects list of text prompts; we give 1 prompt.
        enc = self.processor(text=[self.prompt], images=image, return_tensors="pt")

        item = {}
        for k, v in enc.items():
            if torch.is_tensor(v):
                item[k] = v.squeeze(0)
            else:
                # rarely needed; keep safe
                item[k] = v[0]

        item["target_boxes"] = (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        )
        item["target_labels"] = (
            torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        )
        return item


def collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = None
    if "attention_mask" in batch[0]:
        attention_mask = torch.stack([b["attention_mask"] for b in batch])

    targets = [{"boxes": b["target_boxes"], "labels": b["target_labels"]} for b in batch]

    out = {"pixel_values": pixel_values, "input_ids": input_ids, "targets": targets}
    if attention_mask is not None:
        out["attention_mask"] = attention_mask
    return out


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            # list of targets
            out[k] = [{kk: vv.to(device) for kk, vv in t.items()} for t in v]
    return out


def freeze_backbone_like(model):
    """
    Best-effort freezing of encoders. Names differ between checkpoints/versions.
    We freeze params that look like vision/text encoders. Head stays trainable.
    """
    for name, p in model.named_parameters():
        n = name.lower()
        if any(x in n for x in ["vision_model", "text_model", "backbone", "encoder", "vision", "text"]):
            p.requires_grad = False


def find_logits_and_boxes(outputs):
    """
    OWLv2 outputs fields can differ by transformers version.
    We try to extract:
      - logits: (B, Q, 1)  (single prompt)
      - pred_boxes: (B, Q, 4)
    """
    # logits
    logits = None
    for key in ["logits", "pred_logits", "class_logits"]:
        if hasattr(outputs, key):
            logits = getattr(outputs, key)
            break
        if isinstance(outputs, dict) and key in outputs:
            logits = outputs[key]
            break
    if logits is None:
        raise RuntimeError(f"Cannot find logits in outputs. Available: {dir(outputs)}")

    # boxes
    pred_boxes = None
    for key in ["pred_boxes", "boxes", "predicted_boxes"]:
        if hasattr(outputs, key):
            pred_boxes = getattr(outputs, key)
            break
        if isinstance(outputs, dict) and key in outputs:
            pred_boxes = outputs[key]
            break
    if pred_boxes is None:
        raise RuntimeError(f"Cannot find pred_boxes in outputs. Available: {dir(outputs)}")

    return logits, pred_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--checkpoint", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--prompts_yaml", default="configs/prompts.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/owlvit_ft")
    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze vision/text encoders")
    ap.add_argument("--pred_box_format", default="cxcywh", choices=["cxcywh", "xyxy"])
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)

    # loss weights
    ap.add_argument("--l1_weight", type=float, default=5.0)
    ap.add_argument("--giou_weight", type=float, default=2.0)
    ap.add_argument("--cls_weight", type=float, default=1.0)

    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    meta = json.loads((coco_root / "meta.json").read_text(encoding="utf-8"))
    train_images = Path(meta["train_images_dir"])
    val_images = Path(meta["val_images_dir"])
    train_ann = coco_root / "annotations_train.json"
    val_ann = coco_root / "annotations_val.json"

    prompts = load_prompts(Path(args.prompts_yaml))
    prompt = prompts[0] if prompts else "kidney stone"  # single prompt / single class

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.checkpoint).to(device)

    if args.freeze_backbone:
        freeze_backbone_like(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("All params are frozen. Remove --freeze_backbone or adjust freezing rules.")

    train_ds = CocoForOpenVocab(train_images, train_ann, processor, prompt)
    val_ds = CocoForOpenVocab(val_images, val_ann, processor, prompt)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    writer.add_text("config/checkpoint", args.checkpoint, 0)
    writer.add_text("config/prompt", prompt, 0)
    writer.add_text("config/freeze_backbone", str(args.freeze_backbone), 0)

    best = float("inf")
    best_path = out_dir / "best.pt"
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0, "matched_per_batch": 0.0}

        for batch in tqdm(train_dl, desc=f"train {epoch}/{args.epochs}"):
            batch = to_device(batch, device)

            inputs = {
                "pixel_values": batch["pixel_values"],
                "input_ids": batch["input_ids"],
            }
            if "attention_mask" in batch:
                inputs["attention_mask"] = batch["attention_mask"]

            outputs = model(**inputs)
            logits, pred_boxes = find_logits_and_boxes(outputs)

            # Expect single prompt => logits (B,Q,1) or (B,Q) -> normalize shape
            if logits.dim() == 2:
                logits = logits.unsqueeze(-1)

            loss, stats = compute_detr_style_loss(
                logits=logits,
                pred_boxes=pred_boxes,
                targets=batch["targets"],
                pred_box_format=args.pred_box_format,
                cls_weight=args.cls_weight,
                l1_weight=args.l1_weight,
                giou_weight=args.giou_weight,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
            optim.step()

            for k in run:
                run[k] += stats.get(k, 0.0)

            writer.add_scalar("train/loss_total_step", stats["loss_total"], step)
            writer.add_scalar("train/loss_cls_step", stats["loss_cls"], step)
            writer.add_scalar("train/loss_l1_step", stats["loss_l1"], step)
            writer.add_scalar("train/loss_giou_step", stats["loss_giou"], step)
            writer.add_scalar("train/matched_per_batch_step", stats["matched_per_batch"], step)
            step += 1

        denom = max(1, len(train_dl))
        train_metrics = {k: run[k] / denom for k in run}
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}_epoch", v, epoch)

        # -------- Validation --------
        model.eval()
        vrun = {"loss_total": 0.0, "loss_cls": 0.0, "loss_l1": 0.0, "loss_giou": 0.0, "matched_per_batch": 0.0}
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val", leave=False):
                batch = to_device(batch, device)
                inputs = {
                    "pixel_values": batch["pixel_values"],
                    "input_ids": batch["input_ids"],
                }
                if "attention_mask" in batch:
                    inputs["attention_mask"] = batch["attention_mask"]

                outputs = model(**inputs)
                logits, pred_boxes = find_logits_and_boxes(outputs)
                if logits.dim() == 2:
                    logits = logits.unsqueeze(-1)

                loss, stats = compute_detr_style_loss(
                    logits=logits,
                    pred_boxes=pred_boxes,
                    targets=batch["targets"],
                    pred_box_format=args.pred_box_format,
                    cls_weight=args.cls_weight,
                    l1_weight=args.l1_weight,
                    giou_weight=args.giou_weight,
                )
                for k in vrun:
                    vrun[k] += stats.get(k, 0.0)

        vdenom = max(1, len(val_dl))
        val_metrics = {k: vrun[k] / vdenom for k in vrun}
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}_epoch", v, epoch)

        val_loss = val_metrics["loss_total"]
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"val_loss={val_metrics['loss_total']:.4f} "
            f"best={best:.4f} "
            f"matched/bs={val_metrics['matched_per_batch']:.2f}"
        )

    writer.close()
    print("Saved best:", best_path)


if __name__ == "__main__":
    main()
