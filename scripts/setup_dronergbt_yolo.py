"""
Restructure DroneRGBT flat dirs into YOLO-compatible images/ + labels/ layout.

Before:  DroneRGBT/rgb/train/1.jpg, 1.txt, 100.jpg, 100.txt, ...
After:   DroneRGBT/rgb/images/train/1.jpg, ...
         DroneRGBT/rgb/labels/train/1.txt, ...

YOLO's label auto-discovery replaces '/images/' with '/labels/' in the path,
so this structure is required for labels to be found during training.

Run on Colab BEFORE training:
    python scripts/setup_dronergbt_yolo.py --data_root /content/SGGF-Net/data
"""

import argparse
import shutil
from pathlib import Path


def restructure(modality_dir: Path):
    """Move .jpg → images/{split}/, .txt → labels/{split}/"""
    for split in ("train", "val", "test"):
        src = modality_dir / split
        if not src.exists():
            continue

        imgs_dir = modality_dir / "images" / split
        lbls_dir = modality_dir / "labels" / split
        imgs_dir.mkdir(parents=True, exist_ok=True)
        lbls_dir.mkdir(parents=True, exist_ok=True)

        moved_img, moved_lbl = 0, 0
        for f in src.iterdir():
            if f.suffix in (".jpg", ".jpeg", ".png"):
                shutil.move(str(f), str(imgs_dir / f.name))
                moved_img += 1
            elif f.suffix == ".txt":
                shutil.move(str(f), str(lbls_dir / f.name))
                moved_lbl += 1

        print(f"  {modality_dir.name}/{split}: {moved_img} images, {moved_lbl} labels")

        # Remove empty source dir
        if src.exists() and not any(src.iterdir()):
            src.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/content/SGGF-Net/data")
    args = parser.parse_args()

    data = Path(args.data_root)
    drone = data / "DroneRGBT"

    if not drone.exists():
        print(f"DroneRGBT not found at {drone}")
        return

    for modality in ("thermal", "rgb"):
        mod_dir = drone / modality
        if not mod_dir.exists():
            continue
        # Skip if already restructured
        if (mod_dir / "images").exists():
            print(f"  {modality}: already restructured, skipping")
            continue
        print(f"Restructuring {modality}...")
        restructure(mod_dir)

    print("Done! DroneRGBT is now YOLO-compatible.")


if __name__ == "__main__":
    main()
