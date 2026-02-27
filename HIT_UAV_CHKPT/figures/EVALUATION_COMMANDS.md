# HIT-UAV Figures (No Background Class)

## Regenerate all figures

```bash
cd /Users/harishshankar/Downloads/Project/sggf_net
python scripts/generate_hituav_figures.py \
  --model HIT_UAV_CHKPT/weights/best.pt \
  --data data/hit-uav-2class/hituav_2class.yaml \
  --split val \
  --output_dir HIT_UAV_CHKPT/figures
```

## Output figures

- **confusion_matrix.png** / **confusion_matrix_normalized.png** – 2×2 only (Person, Vehicle), no background
- **BoxPR_curve.png** – Precision-Recall curve
- **BoxF1_curve.png** – F1 vs Confidence
- **BoxP_curve.png** – Precision vs Confidence
- **BoxR_curve.png** – Recall vs Confidence
- **map_vs_iou.png** – mAP vs IoU threshold
- **metrics_bar.png** – Per-class Precision, Recall, AP50
- **detection_examples.png** – Sample predictions
- **EVAL_RESULTS.txt** – Metrics summary
