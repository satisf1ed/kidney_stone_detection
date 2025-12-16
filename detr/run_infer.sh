/home/adzhogov/project/project_v2/.venv/bin/python detr/infer.py \
    --data_yaml "data/data.yaml" \
    --weights runs/detr/best.pt \
    --n 10 \
    --conf 0.25 \
    --device cuda:0