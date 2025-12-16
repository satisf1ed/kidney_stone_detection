import argparse, json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm

from transformers import DetrImageProcessor, DetrForObjectDetection


def set_train_mode(model: DetrForObjectDetection, mode: str) -> None:
    mode = mode.lower().strip()
    if mode not in {"full", "freeze_backbone", "head_only"}:
        raise ValueError(f"Unknown train_mode={mode}. Use full|freeze_backbone|head_only")

    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    if mode == "freeze_backbone":
        for name, p in model.named_parameters():
            if name.startswith("model.backbone") or name.startswith("model.model.backbone"):
                continue
            p.requires_grad = True
        return

    for p in model.class_labels_classifier.parameters():
        p.requires_grad = True
    for p in model.bbox_predictor.parameters():
        p.requires_grad = True


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class CocoDetDataset(Dataset):
    def __init__(self, images_root: Path, ann_json: Path, processor):
        coco = json.loads(ann_json.read_text(encoding="utf-8"))
        self.images_root = images_root
        self.processor = processor
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
        path = self.images_root / info["file_name"]
        image = Image.open(path).convert("RGB")

        anns = self.anns_by_img.get(img_id, [])
        anns = [dict(a, category_id=int(a["category_id"]) - 1) for a in anns]
        target = {"image_id": img_id, "annotations": anns}

        enc = self.processor(images=image, annotations=target, return_tensors="pt")

        item = {}
        for k, v in enc.items():
            if torch.is_tensor(v):
                item[k] = v.squeeze(0)
            else:
                item[k] = v[0]
        return item



def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = [{kk: vv.to(device) for kk, vv in t.items()} for t in v]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--model", default="facebook/detr-resnet-50")
    ap.add_argument(
        "--train_mode",
        default="head_only",
        choices=["head_only", "freeze_backbone", "full"],
        help="Which parts of DETR to train: head_only (default), freeze_backbone, or full",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Max grad norm (0 to disable)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default="runs/detr")
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    meta = json.loads((coco_root / "meta.json").read_text(encoding="utf-8"))
    train_images = Path(meta["train_images_dir"])
    val_images = Path(meta["val_images_dir"])
    train_ann = coco_root / "annotations_train.json"
    val_ann = coco_root / "annotations_val.json"

    device = torch.device(args.device)
    processor = DetrImageProcessor.from_pretrained(args.model)
    model = DetrForObjectDetection.from_pretrained(
        args.model,
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to(device)

    set_train_mode(model, args.train_mode)
    total_p, trainable_p = count_trainable_params(model)
    print(f"DETR train_mode={args.train_mode} trainable_params={trainable_p:,} / total_params={total_p:,}")

    train_ds = CocoDetDataset(train_images, train_ann, processor)
    val_ds = CocoDetDataset(val_images, val_ann, processor)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    writer.add_text("config/train_mode", args.train_mode, 0)
    writer.add_scalar("config/trainable_params", float(trainable_p), 0)
    writer.add_scalar("config/total_params", float(total_p), 0)

    best = float("inf")
    best_path = out_dir / "best.pt"
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        pbar = tqdm(train_dl, desc=f"train {epoch}/{args.epochs}")
        for batch in pbar:
            batch = to_device(batch, device)
            out = model(**batch)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
            optim.step()
            run_loss += loss.item()
            writer.add_scalar("train/loss_step", loss.item(), step)
            step += 1
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = run_loss / max(1, len(train_dl))
        writer.add_scalar("train/loss_epoch", train_loss, epoch)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val", leave=False):
                batch = to_device(batch, device)
                out = model(**batch)
                vloss += out.loss.item()
        vloss /= max(1, len(val_dl))
        writer.add_scalar("val/loss_epoch", vloss, epoch)

        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch}: train={train_loss:.4f} val={vloss:.4f} best={best:.4f}")

    writer.close()
    print("Saved best:", best_path)


if __name__ == "__main__":
    main()
