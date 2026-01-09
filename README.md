# SGGF-Net: UAV Image Object Detection

**UAV Image Object Detection based on Self-Attention Guidance and Global Feature Fusion**

Implementation of SGGF-Net for detecting objects in UAV (drone) images, optimized for training on Apple Silicon (M1/M2) in ~60 minutes.

## Quick Start

### Option 1: Google Colab (Recommended - T4 GPU, ~35-45 min)

1. **Open the Colab notebook**: `SGGF_Net_Training_Colab.ipynb`
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
3. **Run all cells**: They will automatically mount Drive, clone repo, install dependencies
4. **Checkpoints saved to Google Drive**: `/content/drive/MyDrive/SGGF-Net-checkpoints/`

**Total training time: ~35-45 minutes on T4 GPU**

### Option 2: Local M1 (CPU only - ~2-3 hours)

⚠ **Note**: MPS (Apple GPU) crashes due to memory issues. CPU is stable but slow.

### 1. Setup Environment

```bash
# Install dependencies
pip install torch torchvision numpy pillow opencv-python tqdm matplotlib scipy
```

### 2. Prepare Dataset

Ensure HIT-UAV dataset is in `data/hit-uav/` with structure:
```
data/hit-uav/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 3. Run 3-Stage Training

**Stage 1: Baseline (20-25 min)**
```bash
python scripts/train.py --stage 1 --data_dir data/hit-uav --num_classes 6
```

**Stage 2: Enable GFEM (20 min)**
```bash
python scripts/train.py --stage 2 --data_dir data/hit-uav --num_classes 6 \
    --resume checkpoints/stage1_best.pth
```

**Stage 3: Enable NDPA+ARPM (15 min)**
```bash
python scripts/train.py --stage 3 --data_dir data/hit-uav --num_classes 6 \
    --resume checkpoints/stage2_best.pth
```

**Or use the notebook:**
```bash
jupyter notebook SGGF_Net_Training.ipynb
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --checkpoint checkpoints/stage3_best.pth \
    --num_classes 6 \
    --split test
```

## Training Strategy (Optimized for M1)

### Optimizations Applied

1. **Transfer Learning**: Frozen ResNet early layers (layer0, layer1, layer2)
2. **Dataset Subset**: 35% of training data (for research purposes)
3. **Reduced Resolution**: 640×640 (instead of 1536×1536)
4. **Mixed Precision**: FP16 on MPS
5. **Reduced Proposals**: 300 RPN proposals (instead of 1000)
6. **Staged Training**: Progressive enablement of components

### Training Stages

**Stage 1: Baseline Faster-RCNN (8 epochs, ~20-25 min)**
- Train: Backbone (layer3, layer4) + FPN + RPN + ROI Head
- Frozen: GFEM, NDPA, ARPM, ResNet early layers
- Purpose: Stable anchor learning

**Stage 2: Enable GFEM (6 epochs, ~20 min)**
- Train: GFEM module only
- Frozen: Everything else
- Purpose: Learn global feature extraction

**Stage 3: Enable NDPA+ARPM (4 epochs, ~15 min)**
- Train: NDPA and ARPM modules
- Frozen: Everything else
- Purpose: Fine-tune attention modules

**Total Time:**
- **Colab T4 GPU**: ~35-45 minutes (recommended)
- **Local M1 CPU**: ~2-3 hours (stable but slow)
- **Local M1 MPS**: Crashes (not usable)

## Model Architecture

### Components

1. **GFEM (Global Feature Extraction Module)**
   - Transformer-based self-attention
   - Captures long-range dependencies
   - Patch size: 32×32 (optimized)

2. **NDPA (Normal Distribution-based Prior Assigner)**
   - Models bounding boxes as 2D normal distributions
   - Uses KL divergence for matching
   - Improves small object detection

3. **ARPM (Attention-guided ROI Pooling Module)**
   - Multi-scale feature fusion
   - Self-attention enhancement
   - Optimizes ROI features

## Configuration

Default settings (M1 optimized):
- **Device**: CPU (MPS disabled due to memory crashes - see note below)
- **Batch size**: 1
- **Image size**: 640×640
- **Optimizer**: AdamW (lr=1e-4)
- **Mixed precision**: Disabled (CPU/MPS limitations)
- **Dataset subset**: 35% (for faster training)
- **RPN proposals**: 300 (post-NMS)

**⚠ Important: MPS (Apple Silicon GPU) Note:**
- MPS is **disabled by default** due to memory allocation crashes (`IOGPUDeviceShmem` errors)
- The model is too complex for M1's GPU shared memory constraints
- CPU training is slower (~2-3 hours) but **stable and reliable**
- To force MPS (at your own risk): `USE_MPS=1 python scripts/train.py ...`

## Evaluation Metrics

- **mAP**: Mean Average Precision (IoU: 0.5:0.95)
- **AP50**: Average Precision at IoU=0.5
- **Precision, Recall, F1**: Standard detection metrics

## Expected Results

With the optimized training strategy:
- Training completes in ~60 minutes on M1
- mAP will be lower than full training but pipeline is validated
- For research: "Results obtained under hardware constraints (Apple M1)"

## Project Structure

```
sggf_net/
├── models/              # Model architectures
│   ├── gfem.py         # Global Feature Extraction Module
│   ├── ndpa.py         # Normal Distribution-based Prior Assigner
│   ├── arpm.py         # Attention-guided ROI Pooling Module
│   └── sggf_net.py     # Main SGGF-Net architecture
├── utils/               # Utility functions
│   ├── dataset.py      # Dataset loaders
│   ├── transforms.py   # Data augmentation
│   └── metrics.py      # Evaluation metrics
├── scripts/            # Training and evaluation scripts
│   ├── train.py        # Staged training script
│   └── evaluate.py     # Evaluation script
├── checkpoints/        # Saved model checkpoints
├── SGGF_Net_Training.ipynb  # Training notebook
└── requirements.txt    # Python dependencies
```

## Hardware Requirements

- **Recommended**: Apple Silicon (M1/M2) on **CPU mode** (stable)
- **Note**: MPS (GPU) is disabled by default due to memory crashes on M1
- **Alternative**: CUDA GPU (if available, faster)
- **RAM**: 8GB+ recommended
- **Storage**: ~5GB for dataset + checkpoints
- **Training Time**: ~2-3 hours on CPU (M1), ~1 hour on CUDA GPU

## Citation

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
