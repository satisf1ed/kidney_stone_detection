/home/adzhogov/project/project_v2/.venv/bin/python detr/eval.py \
    --coco_root data/coco \
    --split test \
    --weights runs/detr/best.pt \
    --device cuda:0 \
    --conf 0.45