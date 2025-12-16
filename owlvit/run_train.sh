/home/adzhogov/project/project_v2/.venv/bin/python -m owlvit.train \
    --coco_root data/coco \
    --checkpoint google/owlv2-base-patch16-ensemble \
    --prompts_yaml configs/prompts.yaml \
    --epochs 10 --batch 32 --lr 1e-5 \
    --freeze_backbone \
    --device cuda:0