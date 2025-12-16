python compare/roc_auc.py \
  --data data/data.yaml \
  --weights runs/yolo11/train3/weights/best.pt \
  --split test \
  --device 0 \
  --out runs/bench/roc_yolo_test_with_human.png \
  --plot_human \
  --save_scores
