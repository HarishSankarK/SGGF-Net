"""
Pseudo-label Vehicle class for DroneRGBT RGB images.

DroneRGBT has only Person (class 0) annotations.  This script runs a
COCO-pretrained YOLOv11 model on every RGB image and appends high-confidence
Vehicle detections (COCO car/truck/bus/motorcycle → class 1) to the existing
YOLO label files.

Usage (on Colab after cloning repo and mounting Drive):
    python3 scripts/pseudo_label_vehicles.py \
        --rgb_dir data/DroneRGBT/rgb \
        --conf 0.35 \
        --model yolo11x.pt

Splits processed: train, val, test
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# COCO class IDs that map to our "Vehicle" class (1)
VEHICLE_COCO_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck


def run(rgb_dir: Path, conf: float, model_name: str, dry_run: bool):
    from ultralytics import YOLO
    model = YOLO(model_name)   # downloads automatically if not cached

    splits = ["train", "val", "test"]
    total_added = 0

    for split in splits:
        img_dir = rgb_dir / split
        if not img_dir.exists():
            print(f"  [skip] {img_dir} not found")
            continue

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"\n[{split}]  {len(images)} images")

        added_split = 0
        for img_path in tqdm(images, desc=split):
            label_path = img_path.with_suffix(".txt")

            # Run inference at low conf to get all candidates; filter later
            results = model(str(img_path), conf=conf, verbose=False)[0]

            new_lines = []
            if results.boxes is not None:
                for box in results.boxes:
                    cid = int(box.cls[0])
                    if cid not in VEHICLE_COCO_IDS:
                        continue
                    x, y, w, h = box.xywhn[0].tolist()
                    new_lines.append(f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            if not new_lines:
                continue

            if dry_run:
                print(f"  [dry] {img_path.name}: would add {len(new_lines)} vehicle box(es)")
                continue

            # Read existing labels (Person annotations), append Vehicle
            existing = label_path.read_text().strip() if label_path.exists() else ""
            parts = [existing] if existing else []
            parts.extend(new_lines)
            label_path.write_text("\n".join(parts) + "\n")
            added_split += len(new_lines)

        print(f"  Added {added_split} vehicle pseudo-labels in [{split}]")
        total_added += added_split

    print(f"\nDone. Total vehicle pseudo-labels added: {total_added}")


def main():
    p = argparse.ArgumentParser(description="Pseudo-label Vehicle class in DroneRGBT RGB split")
    p.add_argument("--rgb_dir", default="data/DroneRGBT/rgb",
                   help="Path to DroneRGBT/rgb/ folder (contains train/val/test)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Minimum COCO confidence to accept a vehicle detection (default 0.35)")
    p.add_argument("--model", default="yolo11x.pt",
                   help="COCO-pretrained YOLO model to use (default: yolo11x.pt)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be added without writing files")
    args = p.parse_args()

    rgb_dir = Path(args.rgb_dir)
    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgb_dir not found: {rgb_dir}")

    run(rgb_dir, args.conf, args.model, args.dry_run)


if __name__ == "__main__":
    main()
