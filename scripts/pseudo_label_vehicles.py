"""
Pseudo-label Vehicle class for DroneRGBT RGB + Thermal images.

DroneRGBT has only Person (class 0) annotations — no Vehicle labels.

Strategy:
  1. RGB   → run COCO-pretrained YOLOv11 on RGB images; append Vehicle (class 1)
             boxes to existing label files.
  2. Thermal → DroneRGBT is a PAIRED dataset (same scene, co-mounted cameras).
              COCO-pretrained models don't work on thermal, so we copy the vehicle
              boxes from the corresponding RGB label file into the thermal label file.
              Boxes are in normalised xywh, so they transfer well between paired frames.

Usage (on Colab after cloning repo and mounting Drive):
    python3 scripts/pseudo_label_vehicles.py \
        --dronergbt_dir data/DroneRGBT \
        --conf 0.35 \
        --model yolo11x.pt

Splits processed: train, val, test
"""

import argparse
from pathlib import Path
from tqdm import tqdm

# COCO class IDs that map to our "Vehicle" class (1)
VEHICLE_COCO_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck


# ── Step 1: Pseudo-label RGB images via COCO model ───────────────────────────

def label_rgb(rgb_dir: Path, conf: float, model_name: str, dry_run: bool):
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
        print(f"\n[RGB/{split}]  {len(images)} images")

        added = 0
        for img_path in tqdm(images, desc=f"RGB/{split}"):
            label_path = img_path.with_suffix(".txt")

            results = model(str(img_path), conf=conf, verbose=False)[0]

            new_lines = []
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls[0]) not in VEHICLE_COCO_IDS:
                        continue
                    x, y, w, h = box.xywhn[0].tolist()
                    new_lines.append(f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            if not new_lines:
                continue

            if dry_run:
                print(f"  [dry] {img_path.name}: +{len(new_lines)} vehicle")
                continue

            existing = label_path.read_text().strip() if label_path.exists() else ""
            parts = [existing] if existing else []
            parts.extend(new_lines)
            label_path.write_text("\n".join(parts) + "\n")
            added += len(new_lines)

        print(f"  Added {added} vehicle pseudo-labels  [RGB/{split}]")
        total_added += added

    return total_added


# ── Step 2: Copy vehicle lines from RGB labels → paired Thermal labels ────────

def transfer_to_thermal(rgb_dir: Path, thermal_dir: Path, dry_run: bool):
    """
    For each RGB label file that has Vehicle (class 1) lines,
    copy those lines to the paired Thermal label file.
    Images are paired by filename (same stem, e.g. 0001.txt).
    """
    splits = ["train", "val", "test"]
    total_transferred = 0

    for split in splits:
        rgb_lbl_dir = rgb_dir / split
        thr_lbl_dir = thermal_dir / split

        if not rgb_lbl_dir.exists() or not thr_lbl_dir.exists():
            print(f"  [skip] {split}: RGB={rgb_lbl_dir.exists()} Thermal={thr_lbl_dir.exists()}")
            continue

        rgb_labels = sorted(rgb_lbl_dir.glob("*.txt"))
        print(f"\n[Thermal/{split}]  checking {len(rgb_labels)} RGB label files")

        transferred = 0
        for rgb_lbl in tqdm(rgb_labels, desc=f"Thermal/{split}"):
            # Extract vehicle lines from RGB label
            vehicle_lines = [
                ln for ln in rgb_lbl.read_text().splitlines()
                if ln.strip().startswith("1 ")
            ]
            if not vehicle_lines:
                continue

            thr_lbl = thr_lbl_dir / rgb_lbl.name
            if dry_run:
                print(f"  [dry] {rgb_lbl.name}: would copy {len(vehicle_lines)} vehicle line(s) to thermal")
                continue

            existing = thr_lbl.read_text().strip() if thr_lbl.exists() else ""
            parts = [existing] if existing else []
            parts.extend(vehicle_lines)
            thr_lbl.write_text("\n".join(parts) + "\n")
            transferred += len(vehicle_lines)

        print(f"  Transferred {transferred} vehicle lines to Thermal labels  [{split}]")
        total_transferred += transferred

    return total_transferred


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Pseudo-label Vehicle class for DroneRGBT RGB and Thermal"
    )
    p.add_argument("--dronergbt_dir", default="data/DroneRGBT",
                   help="Root of DroneRGBT dataset (contains rgb/ and thermal/)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Min COCO confidence for vehicle detection on RGB (default 0.35)")
    p.add_argument("--model", default="yolo11x.pt",
                   help="COCO-pretrained YOLO weights (default: yolo11x.pt)")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be written without modifying any files")
    p.add_argument("--skip_rgb", action="store_true",
                   help="Skip RGB labeling (useful if already done; go straight to thermal transfer)")
    args = p.parse_args()

    root = Path(args.dronergbt_dir)
    rgb_dir     = root / "rgb"
    thermal_dir = root / "thermal"

    if not root.exists():
        raise FileNotFoundError(f"DroneRGBT root not found: {root}")

    print("\n" + "=" * 55)
    print("  Step 1: Pseudo-label Vehicles in RGB images")
    print("=" * 55)
    if args.skip_rgb:
        print("  [skipped via --skip_rgb]")
        rgb_added = 0
    else:
        rgb_added = label_rgb(rgb_dir, args.conf, args.model, args.dry_run)

    print("\n" + "=" * 55)
    print("  Step 2: Transfer Vehicle labels RGB → Thermal")
    print("=" * 55)
    thr_transferred = transfer_to_thermal(rgb_dir, thermal_dir, args.dry_run)

    print("\n" + "=" * 55)
    print(f"  RGB vehicle labels added    : {rgb_added}")
    print(f"  Thermal labels transferred  : {thr_transferred}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
