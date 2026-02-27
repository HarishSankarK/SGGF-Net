"""
Remap HIT-UAV labels from 5 classes to 2 classes for YOLOv8 validation.
Original: 0=Person, 1=Car, 2=Bicycle, 3=OtherVehicle, 4=DontCare
Mapped:   0=Person, 1=Vehicle (Car,Bicycle,OtherVehicle), DontCare skipped

Creates data/hit-uav-2class/ with remapped labels and symlinked images.
"""
import os
import argparse

def remap_label(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    if cls == 4:  # DontCare - skip
        return None
    if cls in (1, 2, 3):  # Car, Bicycle, OtherVehicle -> vehicle (1)
        cls = 1
    return f"{cls} " + " ".join(parts[1:]) + "\n"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/hit-uav", help="Base HIT-UAV dataset path")
    p.add_argument("--out", default="data/hit-uav-2class", help="Output 2-class dataset path")
    args = p.parse_args()

    src_labels = os.path.join(args.base, "labels")
    out_labels = os.path.join(args.out, "labels")
    src_images = os.path.join(args.base, "images")
    out_images = os.path.join(args.out, "images")

    for split in ("train", "val", "test"):
        src_dir = os.path.join(src_labels, split)
        dst_dir = os.path.join(out_labels, split)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if not f.endswith(".txt"):
                continue
            with open(os.path.join(src_dir, f)) as rf:
                lines = rf.readlines()
            remapped = [ln for ln in (remap_label(ln) for ln in lines) if ln is not None]
            with open(os.path.join(dst_dir, f), "w") as wf:
                wf.writelines(remapped)
        print(f"  labels/{split}: {len([x for x in os.listdir(dst_dir) if x.endswith('.txt')])} files")

    # Hardlink image files (YOLOv8 derives label path from image path - must stay under out/)
    os.makedirs(out_images, exist_ok=True)
    for split in ("train", "val", "test"):
        src_dir = os.path.join(args.base, "images", split)
        dst_dir = os.path.join(out_images, split)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            src_f = os.path.join(src_dir, f)
            dst_f = os.path.join(dst_dir, f)
            if os.path.exists(dst_f):
                continue
            try:
                os.link(src_f, dst_f)
            except OSError:
                import shutil
                shutil.copy2(src_f, dst_f)
        n = len([x for x in os.listdir(dst_dir) if x.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        print(f"  images/{split}: {n} files (hardlinked)")

    print(f"Done. Use: yolo val model=HIT_UAV_CHKPT/weights/best.pt data=data/hit-uav-2class/hituav_2class.yaml")

if __name__ == "__main__":
    main()
