/home/adzhogov/project/project_v2/.venv/bin/python detr/train.py \
    --coco_root data/coco \
    --epochs 10 \
    --batch 32 \
    --lr 1e-4 \
    --device cuda:0 \
    --train_mode freeze_backbone