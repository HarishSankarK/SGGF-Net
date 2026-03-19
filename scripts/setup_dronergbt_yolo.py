"""
Restructure DroneRGBT into YOLO-compatible images/ + labels/ layout.

DroneRGBT comes with:
  - annotations/train/*.txt, annotations/val/*.txt, annotations/test/*.txt  (YOLO labels)
  - rgb/train/*.jpg, rgb/val/*.jpg, rgb/test/*.jpg                         (RGB images)
  - thermal/train/*.jpg, thermal/val/*.jpg, thermal/test/*.jpg             (Thermal images)

This script creates:
  - rgb/images/train/*.jpg      + rgb/labels/train/*.txt
  - thermal/images/train/*.jpg  + thermal/labels/train/*.txt
  (same labels for both since they're paired images of the same scene)

YOLO's label auto-discovery replaces '/images/' with '/labels/' in the path.

Run on Colab BEFORE training:
    python scripts/setup_dronergbt_yolo.py --data_root /content/SGGF-Net/data
"""

import argparse
import shutil
from pathlib import Path


def setup_modality(modality_dir: Path, annotations_dir: Path):
    """
    Move images into images/{split}/ and copy labels from annotations/ into labels/{split}/.
    Both rgb and thermal share the same annotation files (paired dataset).
    """
    for split in ("train", "val", "test"):
        src_imgs = modality_dir / split
        src_lbls = annotations_dir / split

        if not src_imgs.exists():
            print(f"    [skip] {modality_dir.name}/{split}: image dir not found")
            continue

        imgs_dir = modality_dir / "images" / split
        lbls_dir = modality_dir / "labels" / split
        imgs_dir.mkdir(parents=True, exist_ok=True)
        lbls_dir.mkdir(parents=True, exist_ok=True)

        # Move images (if still in flat dir)
        moved_img = 0
        for f in list(src_imgs.iterdir()):
            if f.suffix in (".jpg", ".jpeg", ".png"):
                shutil.move(str(f), str(imgs_dir / f.name))
                moved_img += 1

        # Copy labels from annotations/ dir
        copied_lbl = 0
        if src_lbls.exists():
            for f in src_lbls.iterdir():
                if f.suffix == ".txt":
                    dest = lbls_dir / f.name
                    if not dest.exists():
                        shutil.copy2(str(f), str(dest))
                        copied_lbl += 1

        # Also move any .txt already in the flat image dir
        moved_lbl = 0
        for f in list(src_imgs.iterdir()):
            if f.suffix == ".txt":
                dest = lbls_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
                    moved_lbl += 1
                else:
                    f.unlink()  # duplicate, remove
                    moved_lbl += 1

        total_imgs = len(list(imgs_dir.glob("*.jpg"))) + len(list(imgs_dir.glob("*.png")))
        total_lbls = len(list(lbls_dir.glob("*.txt")))
        print(f"    {modality_dir.name}/{split}: {total_imgs} images, {total_lbls} labels")

        # Remove empty source dir
        if src_imgs.exists() and not any(src_imgs.iterdir()):
            src_imgs.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/SGGF-Net/data")
    args = parser.parse_args()

    data = Path(args.data_root)
    drone = data / "DroneRGBT"
    annotations = drone / "annotations"

    if not drone.exists():
        print(f"DroneRGBT not found at {drone}")
        return

    if not annotations.exists():
        print(f"WARNING: annotations/ dir not found at {annotations}")
        print("Labels may not be copied correctly.")

    for modality in ("thermal", "rgb"):
        mod_dir = drone / modality
        if not mod_dir.exists():
            continue
        # Skip if already restructured with correct label count
        if (mod_dir / "images").exists() and (mod_dir / "labels").exists():
            train_imgs = len(list((mod_dir / "images" / "train").glob("*.jpg")))
            train_lbls = len(list((mod_dir / "labels" / "train").glob("*.txt")))
            if train_imgs > 0 and train_lbls >= train_imgs * 0.9:
                print(f"  {modality}: already set up ({train_imgs} images, {train_lbls} labels), skipping")
                continue
            else:
                print(f"  {modality}: incomplete ({train_imgs} images, {train_lbls} labels), re-running...")

        print(f"  Setting up {modality}...")
        setup_modality(mod_dir, annotations)

    print("\nDone! DroneRGBT is now YOLO-compatible.")
    print("Verify: labels count should match images count for each split.")


if __name__ == "__main__":
    main()
