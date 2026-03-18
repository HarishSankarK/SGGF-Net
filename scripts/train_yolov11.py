"""
YOLOv11 Training / Finetuning Script
=====================================
Supports: HIT-UAV | DroneRGBT | Combined

Usage examples:
  # Train from scratch on HIT-UAV
  python scripts/train_yolov11.py --dataset hituav

  # Train from scratch on DroneRGBT
  python scripts/train_yolov11.py --dataset dronergbt

  # Train on combined dataset
  python scripts/train_yolov11.py --dataset combined

  # Finetune from existing checkpoint
  python scripts/train_yolov11.py --dataset hituav --finetune checkpoints/best.pt

  # Finetune HIT_UAV_CHKPT on DroneRGBT
  python scripts/train_yolov11.py --dataset dronergbt --finetune sggf_net/HIT_UAV_CHKPT/weights/best.pt

  # Choose model size: n / s / m / l / x (default: m)
  python scripts/train_yolov11.py --dataset combined --model yolo11m.pt

  # Custom epochs / batch / image size
  python scripts/train_yolov11.py --dataset hituav --epochs 100 --batch 8 --imgsz 640
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


# ── Dataset YAML paths (relative to project root sggf_net/) ─────────────────
DATASET_YAMLS = {
    "hituav":         "data/hit-uav-2class/hituav_2class.yaml",
    "dronergbt":      "data/dronergbt_yolo11.yaml",
    "combined":       "data/combined_yolo11.yaml",
    "full_combined":  "data/full_combined_yolo11.yaml",   # HIT-UAV + DroneRGBT thermal + RGB
}

# ── Default pretrained YOLOv11 weights (downloaded automatically) ─────────────
DEFAULT_MODEL = "yolo11m.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train / finetune YOLOv11")

    parser.add_argument(
        "--dataset",
        choices=["hituav", "dronergbt", "combined", "full_combined"],
        required=True,
        help="Dataset to train on",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to .pt checkpoint to finetune from (skips pretrained download)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"YOLOv11 model variant or path (default: {DEFAULT_MODEL}). "
             "Ignored if --finetune is set.",
    )
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--workers", type=int,   default=4)
    parser.add_argument("--device",  type=str,   default="",
                        help="Device: '' = auto, 'cpu', '0', '0,1', ...")
    parser.add_argument("--lr0",     type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--freeze",  type=int,   default=None,
                        help="Freeze first N layers (useful for finetuning)")
    parser.add_argument("--project", type=str,   default="checkpoints/yolov11",
                        help="Output directory for runs")
    parser.add_argument("--name",    type=str,   default=None,
                        help="Run name (auto-generated if not set)")
    parser.add_argument("--patience", type=int,  default=50,
                        help="Early stopping patience (0 = disabled)")

    return parser.parse_args()


def resolve_paths(args):
    """Resolve YAML and checkpoint paths relative to the sggf_net/ root."""
    # Locate sggf_net/ directory (parent of this script's directory)
    script_dir = Path(__file__).resolve().parent       # sggf_net/scripts/
    root_dir   = script_dir.parent                     # sggf_net/

    yaml_rel  = DATASET_YAMLS[args.dataset]
    yaml_path = root_dir / yaml_rel

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {yaml_path}\n"
            f"Make sure you ran the appropriate preprocessing script first."
        )

    model_path = args.finetune if args.finetune else args.model

    if args.finetune and not Path(args.finetune).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.finetune}")

    return str(yaml_path), model_path, root_dir


def build_run_name(args):
    if args.name:
        return args.name
    mode = "finetune" if args.finetune else "train"
    return f"yolov11_{args.dataset}_{mode}"


def main():
    args = parse_args()
    yaml_path, model_path, root_dir = resolve_paths(args)
    run_name = build_run_name(args)

    print("\n" + "=" * 60)
    print(f"  YOLOv11 {'Finetuning' if args.finetune else 'Training'}")
    print("=" * 60)
    print(f"  Dataset  : {args.dataset}  ({yaml_path})")
    print(f"  Model    : {model_path}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Img size : {args.imgsz}")
    print(f"  Output   : {args.project}/{run_name}")
    if args.finetune:
        print(f"  Freeze   : first {args.freeze} layers" if args.freeze else "  Freeze   : none (all layers trainable)")
    print("=" * 60 + "\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = YOLO(model_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_kwargs = dict(
        data     = yaml_path,
        epochs   = args.epochs,
        batch    = args.batch,
        imgsz    = args.imgsz,
        workers  = args.workers,
        lr0      = args.lr0,
        project  = str(root_dir / args.project),
        name     = run_name,
        patience = args.patience,
        save     = True,
        plots    = True,
        verbose  = True,
    )

    # Device (empty string = Ultralytics auto-select)
    if args.device:
        train_kwargs["device"] = args.device

    # Freeze layers when finetuning
    if args.finetune and args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    # Lower LR for finetuning if not explicitly overridden
    if args.finetune and args.lr0 == 0.01:
        train_kwargs["lr0"] = 0.001
        train_kwargs["lrf"] = 0.01
        print("  [Finetune] lr0 auto-adjusted to 0.001")

    results = model.train(**train_kwargs)

    # ── Summary ───────────────────────────────────────────────────────────────
    save_dir = Path(results.save_dir)
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best weights : {save_dir / 'weights' / 'best.pt'}")
    print(f"  Last weights : {save_dir / 'weights' / 'last.pt'}")
    print(f"  Results      : {save_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
