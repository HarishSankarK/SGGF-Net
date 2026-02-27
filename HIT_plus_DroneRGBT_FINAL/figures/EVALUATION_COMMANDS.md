# HIT_plus_DroneRGBT_FINAL Evaluation and Figure Generation

Use **`generate_hituav_figures.py`** (same script as HIT_UAV_CHKPT) to produce all figures with 2 classes (Person, Vehicle) and confusion matrix WITHOUT background.

## Validation (HIT-UAV + DroneRGBT combined)

```bash
cd /Users/harishshankar/Downloads/Project/sggf_net

python scripts/generate_hituav_figures.py \
  --model HIT_plus_DroneRGBT_FINAL/weights/best.pt \
  --data data/combined_hit_dronergbt/combined.yaml \
  --split val \
  --output_dir HIT_plus_DroneRGBT_FINAL/figures
```

## Test split (optional)

```bash
python scripts/generate_hituav_figures.py \
  --model HIT_plus_DroneRGBT_FINAL/weights/best.pt \
  --data data/combined_hit_dronergbt/combined.yaml \
  --split test \
  --output_dir HIT_plus_DroneRGBT_FINAL/figures_test
```

## Figures generated (same as HIT_UAV_CHKPT)

- `confusion_matrix.png`, `confusion_matrix.pdf` (2×2 Person, Vehicle only; no background)
- `confusion_matrix_normalized.png`, `confusion_matrix_normalized.pdf`
- `BoxPR_curve.png`, `BoxPR_curve.pdf`
- `BoxF1_curve.png`, `BoxF1_curve.pdf`
- `BoxP_curve.png`, `BoxP_curve.pdf`
- `BoxR_curve.png`, `BoxR_curve.pdf`
- `map_vs_iou.png`, `map_vs_iou.pdf`
- `metrics_bar.png`, `metrics_bar.pdf`
- `detection_examples.png`, `detection_examples.pdf`
- `EVAL_RESULTS.txt`

## Dataset

- **Combined HIT-UAV + DroneRGBT** (2 classes: Person, Vehicle)
- Config: `data/combined_hit_dronergbt/combined.yaml`
- Val images: ~467 (both HIT-UAV and DroneRGBT)
- Test images: ~2372
