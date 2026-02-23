# SGGF-Net / Fusion-YOLOv11: Multimodal UAV Object Detection

**RGB-Thermal object detection for UAV/drone imagery** — Dual-stream SGGF-Net backbone with mid-level fusion and YOLOv11 detection head.

- **Fusion-YOLOv11**: Primary implementation — trains on DroneRGBT (RGB-Thermal), SMOD (RGB), and HIT-UAV (thermal). Classes: **person**, **vehicle**.
- **SGGF-Net**: Original single-modality Faster R-CNN variant (see `scripts/train.py`).

---

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open **`FusionYOLOv11_Colab.ipynb`** (in project root)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells: mount Drive, install deps, clone repo, train
4. Checkpoints save to: `/content/drive/MyDrive/FusionYOLOv11-checkpoints/`

```bash
# Training command (Colab cell)
!python3 scripts/train_fusion_yolov11.py \
  --dataset combined_all \
  --hituav_dir data/hit-uav \
  --dronergbt_dir data/DroneRGBT \
  --smod_dir data/SMOD \
  --checkpoint_dir /content/drive/MyDrive/FusionYOLOv11-checkpoints \
  --stage 1 --max_size 640 --batch_size 8 --num_workers 2
```

### Option 2: Laptop with NVIDIA GPU (4–8GB VRAM)

```bash
# Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Laptop mode (4–6GB): batch=4, AMP, max_size=640
python scripts/train_fusion_yolov11.py --dataset combined_all --laptop \
  --hituav_dir data/hit-uav --dronergbt_dir data/DroneRGBT --smod_dir data/SMOD \
  --checkpoint_dir checkpoints --stage 1

# 8GB+ GPU: omit --laptop (defaults: batch=8, max_size=640)
```

### Option 3: Single Dataset

```bash
# DroneRGBT only (RGB-Thermal, person)
python scripts/train_fusion_yolov11.py --dataset dronergbt --data_dir data/DroneRGBT --stage 1

# SMOD only (RGB, person+vehicle)
python scripts/train_fusion_yolov11.py --dataset smod --data_dir data/SMOD --stage 1

# HIT-UAV only (thermal/infrared, person+vehicle)
python scripts/train_fusion_yolov11.py --dataset hituav --data_dir data/hit-uav --stage 1
```

---

## Datasets

### Supported Datasets

| Dataset     | Modality      | Classes    | Structure                |
|------------|---------------|------------|--------------------------|
| **DroneRGBT** | RGB + Thermal | person     | `rgb/train`, `thermal/train`, `labels/train` |
| **SMOD**   | RGB           | person, vehicle | `images/train`, `labels/train` (YOLO format) |
| **HIT-UAV** | Thermal/Infrared | person, vehicle | `images/train`, `labels/train` (YOLO format) |

### Class Mapping (Unified 2-Class System)

- **Person** → class 1
- **Vehicle** (Car, Bicycle, OtherVehicle) → class 2  
- **DontCare** → skipped

### Directory Layout

```
data/
├── hit-uav/           # HIT-UAV (thermal)
│   ├── images/train, val
│   └── labels/train, val
├── DroneRGBT/         # RGB-Thermal pairs
│   ├── rgb/train, val
│   ├── thermal/train, val
│   └── labels/train, val
└── SMOD/              # RGB
    ├── images/train, val
    └── labels/train, val
```

### Preprocessing

**HIT-UAV** (raw JSON/XML from [HIT-UAV-Infrared-Thermal-Dataset](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)):

```bash
python scripts/preprocess_hituav.py --source data/HIT-UAV-raw --target data/hit-uav
```

**DroneRGBT** (point annotations → YOLO boxes):

```bash
python scripts/preprocess_dronergbt.py --source data/DroneRGBT-raw --target data/DroneRGBT
```

---

## Training

### Script: `train_fusion_yolov11.py`

### Dataset Modes

| Mode           | Datasets                        | Default Ratios          |
|----------------|----------------------------------|-------------------------|
| `combined`     | DroneRGBT + SMOD                | 100% each               |
| `combined_all` | HIT-UAV + DroneRGBT + SMOD      | 100% HIT, 50% DroneRGBT, 25% SMOD |

Override ratios: `--dronergbt_subset_ratio 0.5 --smod_subset_ratio 0.25`

### Stage-by-Stage Training (Recommended)

| Stage | Trainable                                      | Frozen                   | Epochs | LR    |
|-------|------------------------------------------------|--------------------------|--------|-------|
| 1     | Backbone (layer3,4) + Fusion + PANet + Head    | GFEM, backbone layer0–2  | 30     | 5e-5  |
| 2     | GFEM only                                      | Rest                     | 20     | 1e-5  |
| 3     | Full fine-tune                                 | None                     | 30     | 5e-6  |

Chain stages:

```bash
# Stage 1
python scripts/train_fusion_yolov11.py --dataset combined_all --stage 1 \
  --hituav_dir data/hit-uav --dronergbt_dir data/DroneRGBT --smod_dir data/SMOD

# Stage 2
python scripts/train_fusion_yolov11.py --stage 2 --resume checkpoints/best.pth \
  --dataset combined_all --hituav_dir data/hit-uav --dronergbt_dir data/DroneRGBT --smod_dir data/SMOD

# Stage 3
python scripts/train_fusion_yolov11.py --stage 3 --resume checkpoints/best.pth \
  --dataset combined_all --hituav_dir data/hit-uav --dronergbt_dir data/DroneRGBT --smod_dir data/SMOD
```

### Key Arguments

| Argument              | Default | Description                          |
|-----------------------|---------|--------------------------------------|
| `--dataset`           | dronergbt | dronergbt, hituav, smod, combined, combined_all |
| `--stage`             | None    | 1, 2, 3 for stage-wise training      |
| `--batch_size`        | 8       | Reduce for low VRAM                 |
| `--max_size`          | 640     | Image resize max (Colab: 640)        |
| `--laptop`            | -       | batch=4, num_workers=2, use_amp      |
| `--use_amp`           | -       | Mixed precision (FP16)              |
| `--val_conf_threshold`| 0.25    | Lower (e.g. 0.05) for early training |
| `--resume`            | None    | Path or "latest"/"best"              |
| `--auto_resume`       | -       | Resume from latest.pth if present   |

---

## Evaluation

**Validation:**
```bash
python scripts/evaluate_fusion.py \
  --dataset smod \
  --data_dir data/SMOD \
  --checkpoint checkpoints_smod_fresh/best.pth \
  --num_classes 3 \
  --split val
```

**Test:**
```bash
python scripts/evaluate_fusion.py \
  --dataset smod \
  --data_dir data/SMOD \
  --checkpoint checkpoints_smod_fresh/best.pth \
  --num_classes 3 \
  --split test
```

For low-VRAM GPUs: `--laptop`. For CPU-only: `--cpu`.

**Combined/other datasets:** Use `--dataset combined_all` with `--hituav_dir`, `--dronergbt_dir`, `--smod_dir` as needed.

### Figure Generation (for manuscript)

Generates PR curves, confusion matrix, mAP vs IoU, metrics bar chart, detection examples, and training curves:

```bash
python scripts/generate_eval_figures.py \
  --dataset smod \
  --data_dir data/SMOD \
  --checkpoint checkpoints_smod_fresh/best.pth \
  --output_dir ../Paper/figures \
  --split val
```

For test split: `--split test`. Training curves use `training_log.csv` (auto-detected from checkpoint directory). CPU: add `--cpu`.

---

## Model Architecture

**Fusion-YOLOv11** pipeline:

1. **Dual-stream backbone** — GFEM + ResNet50 per modality (RGB, Thermal)
2. **Mid-level fusion** — Concat + cross-modal attention at C2–C5
3. **PANet** — Multi-scale feature aggregation
4. **YOLOv11 head** — Anchor-free detection (objectness + bbox + class logits)

Single-modality input (e.g. HIT-UAV, SMOD): RGB image duplicated as both RGB and thermal streams.

---

## Evaluation Metrics

- **mAP** — Mean AP (IoU 0.5:0.95)
- **AP50** — AP at IoU 0.5
- **Precision, Recall, F1** — Via `evaluate_fusion.py`

---

## Project Structure

```
sggf_net/
├── models/
│   ├── fusion_yolov11.py    # Fusion-YOLOv11 (main model)
│   ├── yolo_head.py        # YOLOv11 head + loss
│   ├── yolo_utils.py       # decode_bbox, post_process, assign_targets
│   ├── gfem.py             # Global Feature Extraction Module
│   ├── fusion.py           # MidLevelFusion
│   ├── panet.py            # PANet
│   ├── sggf_net.py         # ResNet backbone, legacy SGGF-Net
│   └── ...
├── utils/
│   ├── dataset.py          # DroneRGBTDataset, HITUAVDataset, SMODDataset
│   ├── transforms.py       # Resize, Pad, RandomHorizontalFlip
│   └── metrics.py          # mAP, AP50, IoU
├── scripts/
│   ├── train_fusion_yolov11.py   # Main training (Fusion-YOLOv11)
│   ├── evaluate_fusion.py        # Evaluation
│   ├── generate_eval_figures.py  # PR curves, confusion matrix, etc.
│   ├── preprocess_hituav.py      # HIT-UAV preprocessing
│   ├── preprocess_dronergbt.py   # DroneRGBT preprocessing
│   ├── preprocess_smod.py       # SMOD preprocessing
│   ├── train.py                  # Legacy SGGF-Net (HIT-UAV)
│   └── evaluate.py               # Legacy evaluation
├── data/                   # Datasets (hit-uav, DroneRGBT, SMOD)
├── requirements.txt
└── README.md
```

---

## Hardware Requirements

| Platform              | Notes                                                     |
|-----------------------|-----------------------------------------------------------|
| **Colab T4**          | Recommended; max_size 640, batch 8 (~15–25 min/epoch)    |
| **Laptop NVIDIA 4–6GB** | Use `--laptop` (batch 4, AMP)                          |
| **Laptop NVIDIA 8GB+** | Defaults OK (batch 8)                                   |
| **Apple M1/M2**       | Legacy `train.py` CPU only; MPS unstable for Fusion     |
| **RAM**               | 8GB+                                                     |

---

## Requirements

```bash
pip install -r requirements.txt
# For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

From `requirements.txt`: torch, torchvision, numpy, Pillow, tqdm, matplotlib, opencv-python

---

## Citation

Original SGGF-Net:

```bibtex
@article{bai2024uav,
  title={UAV image object detection based on self-attention guidance and global feature fusion},
  author={Bai, Jing and Hu, Haiyang and Liu, Xiaojing and Zhuang, Shanna and Wang, Zhengyou},
  journal={Image and Vision Computing},
  volume={151},
  pages={105262},
  year={2024}
}
```

Fusion-YOLOv11 extends this with RGB-Thermal dual-stream fusion and a YOLOv11 detection head.
