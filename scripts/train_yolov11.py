"""
YOLOv11 Training / Finetuning Script  (Drive-resumable)
=========================================================
Supports: HIT-UAV | DroneRGBT | Combined | Full-Combined

Usage examples:
  # Train from scratch on HIT-UAV
  python scripts/train_yolov11.py --dataset hituav

  # Train on combined dataset, save directly to Google Drive
  python scripts/train_yolov11.py --dataset full_combined \
      --drive_dir /content/drive/MyDrive/sggf_checkpoints/yolov11 \
      --name combined_finetune4

  # Auto-resume from Drive (detects last.pt automatically)
  python scripts/train_yolov11.py --dataset full_combined \
      --drive_dir /content/drive/MyDrive/sggf_checkpoints/yolov11 \
      --name combined_finetune4 --resume

  # Finetune from existing checkpoint
  python scripts/train_yolov11.py --dataset hituav --finetune checkpoints/best.pt

  # Choose model size: n / s / m / l / x (default: m)
  python scripts/train_yolov11.py --dataset combined --model yolo11m.pt

  # Custom epochs / batch / image size
  python scripts/train_yolov11.py --dataset hituav --epochs 100 --batch 8 --imgsz 640

Drive-resume behaviour:
  - If --drive_dir is set, the run project folder is placed INSIDE Drive so
    Ultralytics writes weights directly there (no copy step needed).
  - If --resume is passed AND last.pt exists in Drive, training continues
    from the last saved epoch automatically.
  - If --resume is passed but last.pt is missing, training starts fresh
    with a warning (safe first-run behaviour).
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
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
    parser.add_argument("--drive_dir", type=str, default=None,
                        help="Google Drive folder for checkpoints, e.g. "
                             "/content/drive/MyDrive/sggf_checkpoints/yolov11 "
                             "If set, overrides --project and saves directly to Drive.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last.pt in Drive (requires --drive_dir and --name). "
                             "Safe to pass on first run — falls back to fresh training if "
                             "last.pt is not found.")

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


def resolve_drive(args, root_dir, run_name):
    """
    Determine the project directory and whether we are resuming.

    Returns (project_dir: str, resume_from: str | None)
      - project_dir  : passed as `project=` to model.train()
      - resume_from  : path to last.pt if resuming, else None
    """
    if args.drive_dir:
        project_dir = str(Path(args.drive_dir))
    else:
        project_dir = str(root_dir / args.project)

    resume_from = None
    if args.resume:
        last_pt = Path(project_dir) / run_name / "weights" / "last.pt"
        if last_pt.exists():
            resume_from = str(last_pt)
            print(f"  [Resume] Found last.pt → {resume_from}")
        else:
            print(f"  [Resume] last.pt not found at {last_pt} — starting fresh")

    return project_dir, resume_from


# ── Sample prediction every N epochs ─────────────────────────────────────────

SAMPLE_IMAGES = {
    "HIT-UAV":           "data/hit-uav-2class/images/val",
    "DroneRGBT-RGB":     "data/DroneRGBT/rgb/images/val",
    "DroneRGBT-Thermal": "data/DroneRGBT/thermal/images/val",
}
PREDICT_EVERY = 5  # epochs


def pick_random_samples(root_dir: Path) -> dict:
    """Pick one random image from each dataset source."""
    samples = {}
    for name, rel_path in SAMPLE_IMAGES.items():
        img_dir = root_dir / rel_path
        if not img_dir.exists():
            continue
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if imgs:
            samples[name] = random.choice(imgs)
    return samples


def display_predictions(save_dir: Path, samples: dict, epoch: int):
    """Run inference on sample images and display them inline (Colab/Jupyter)."""
    # Load current best/last weights via YOLO wrapper (not raw nn.Module)
    last_pt = save_dir / "weights" / "last.pt"
    if not last_pt.exists():
        print(f"  [sample predictions] last.pt not found, skipping")
        return

    pred_model = YOLO(str(last_pt))

    try:
        from IPython.display import display, HTML
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except (ImportError, AttributeError):
        in_notebook = False

    pred_dir = save_dir / "sample_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"  Sample predictions — Epoch {epoch}")
    print(f"{'─'*50}")

    for name, img_path in samples.items():
        results = pred_model(str(img_path), conf=0.20, iou=0.30, verbose=False)[0]
        n_det = len(results.boxes) if results.boxes is not None else 0
        annotated = results.plot()

        out_path = pred_dir / f"epoch{epoch:03d}_{name}.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {name}: {n_det} detections  ({img_path.name})")

        if in_notebook:
            from IPython.display import Image, display as ipy_display
            ipy_display(HTML(f"<b>{name}</b> — {n_det} detections (epoch {epoch})"))
            ipy_display(Image(filename=str(out_path), width=500))

    print(f"  Saved to {pred_dir}/")
    print(f"{'─'*50}\n")

    # Free memory
    del pred_model


def make_epoch_end_callback(root_dir: Path, save_dir: Path):
    """Create a callback that predicts samples every PREDICT_EVERY epochs."""
    samples = pick_random_samples(root_dir)

    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1  # 0-indexed → 1-indexed
        if epoch % PREDICT_EVERY != 0:
            return
        display_predictions(save_dir, samples, epoch)

    return on_train_epoch_end


def main():
    args = parse_args()
    yaml_path, model_path, root_dir = resolve_paths(args)
    run_name = build_run_name(args)
    project_dir, resume_from = resolve_drive(args, root_dir, run_name)

    print("\n" + "=" * 60)
    print(f"  YOLOv11 {'Resume' if resume_from else 'Finetune' if args.finetune else 'Training'}")
    print("=" * 60)
    print(f"  Dataset  : {args.dataset}  ({yaml_path})")
    print(f"  Model    : {resume_from if resume_from else model_path}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Img size : {args.imgsz}")
    print(f"  Output   : {project_dir}/{run_name}")
    if args.drive_dir:
        print(f"  Drive    : {args.drive_dir}  (checkpoints saved directly to Drive)")
    if args.finetune and not resume_from:
        print(f"  Freeze   : first {args.freeze} layers" if args.freeze else "  Freeze   : none (all layers trainable)")
    print("=" * 60 + "\n")

    # ── Load model ────────────────────────────────────────────────────────────
    # When resuming, load last.pt; Ultralytics restores all training state
    model = YOLO(resume_from if resume_from else model_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_kwargs = dict(
        data     = yaml_path,
        epochs   = args.epochs,
        batch    = args.batch,
        imgsz    = args.imgsz,
        workers  = args.workers,
        lr0      = args.lr0,
        project  = project_dir,
        name     = run_name,
        patience = args.patience,
        save     = True,
        plots    = True,
        verbose  = True,
    )

    # When resuming, Ultralytics needs resume=True to restore optimizer/epoch state
    if resume_from:
        train_kwargs["resume"] = True

    # Device (empty string = Ultralytics auto-select)
    if args.device:
        train_kwargs["device"] = args.device

    # Freeze layers when finetuning (not applicable when resuming)
    if args.finetune and not resume_from and args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    # Lower LR for finetuning if not explicitly overridden and not resuming
    if args.finetune and not resume_from and args.lr0 == 0.01:
        train_kwargs["lr0"] = 0.001
        train_kwargs["lrf"] = 0.01
        print("  [Finetune] lr0 auto-adjusted to 0.001")

    # ── Register sample-prediction callback ───────────────────────────────────
    save_path = Path(project_dir) / run_name
    callback = make_epoch_end_callback(root_dir, save_path)
    model.add_callback("on_train_epoch_end", callback)

    results = model.train(**train_kwargs)

    # ── Summary ───────────────────────────────────────────────────────────────
    save_dir = Path(results.save_dir)
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best weights : {save_dir / 'weights' / 'best.pt'}")
    print(f"  Last weights : {save_dir / 'weights' / 'last.pt'}")
    print(f"  Results      : {save_dir}")
    if args.drive_dir:
        print(f"  Drive path   : {args.drive_dir}/{run_name}/weights/best.pt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
