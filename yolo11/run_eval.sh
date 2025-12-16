/home/adzhogov/project/project_v2/.venv/bin/python yolo11/eval.py \
    --data "data/data.yaml" \
    --weights runs/yolo11/train3/weights/best.pt \
    --device 0 \
    --split test