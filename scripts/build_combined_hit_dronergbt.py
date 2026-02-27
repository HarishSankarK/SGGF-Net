"""
Build combined HIT-UAV + DroneRGBT dataset for YOLOv8 (2 classes: person, vehicle).
- HIT-UAV: from hit-uav-2class (already 2-class)
- DroneRGBT: thermal images, annotations (class 0=person only; no vehicle)
"""
import os
import shutil
from pathlib import Path

def main():
    base = Path("data")
    hit = base / "hit-uav-2class"
    drone = base / "DroneRGBT"
    out = base / "combined_hit_dronergbt"

    for split in ("val", "test"):
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # HIT-UAV (prefix to avoid collision)
        hit_img = hit / "images" / split
        hit_lbl = hit / "labels" / split
        if hit_img.exists():
            for f in hit_img.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    base_name = f"hituav_{f.stem}"
                    shutil.copy2(f, img_out / f"{base_name}{f.suffix}")
                    lbl_f = hit_lbl / f"{f.stem}.txt"
                    if lbl_f.exists():
                        shutil.copy2(lbl_f, lbl_out / f"{base_name}.txt")

        # DroneRGBT thermal (class 0=person)
        drone_img = drone / "thermal" / split
        drone_ann = drone / "annotations" / split
        if drone_img.exists():
            for f in drone_img.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    base_name = f"drone_{f.stem}"
                    shutil.copy2(f, img_out / f"{base_name}{f.suffix}")
                    ann_f = drone_ann / f"{f.stem}.txt"
                    if ann_f.exists():
                        shutil.copy2(ann_f, lbl_out / f"{base_name}.txt")

        n_img = len(list(img_out.glob("*.*")))
        n_lbl = len(list(lbl_out.glob("*.txt")))
        print(f"  {split}: {n_img} images, {n_lbl} labels")

    print(f"Done. Dataset at {out}")

if __name__ == "__main__":
    main()
