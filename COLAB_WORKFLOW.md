# Complete Colab Workflow Guide

## Overview

This guide explains the complete workflow for training SGGF-Net in Google Colab with:
- ✅ Dataset in `data/` folder (organized structure)
- ✅ Checkpoints saved to Google Drive (persistent storage)
- ✅ Resume training capability from Drive checkpoints

## Directory Structure in Colab

```
/content/
├── SGGF-Net/                    # Cloned repository
│   ├── data/
│   │   └── hit-uav/            # Dataset (copied from Drive)
│   ├── models/
│   ├── scripts/
│   └── ...
└── drive/
    └── MyDrive/
        └── SGGF-Net-checkpoints/  # Checkpoints (persistent)
            ├── latest.pth
            └── best.pth
```

## Step-by-Step Workflow

### Step 1: Mount Drive and Clone Repo

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/HarishSankarK/SGGF-Net.git
%cd SGGF-Net
```

### Step 2: Install Dependencies

```python
# Install PyTorch with CUDA
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install -r requirements.txt
```

### Step 3: Verify Dataset

The dataset is **already included** in the GitHub repository at `data/hit-uav/`!

```python
# Dataset comes with the repository - just verify it's there
!ls -la data/hit-uav/
print("✓ Dataset found in repository!")
```

**No upload needed!** The dataset is part of the GitHub repository.

### Step 4: Verify Setup

```python
# Check GPU
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Test dataset
!python scripts/test_hituav_dataset.py
```

### Step 5: Start Training (Checkpoints to Drive)

```python
# Setup Drive checkpoint directory
import os
drive_checkpoint_dir = '/content/drive/MyDrive/SGGF-Net-checkpoints'
os.makedirs(drive_checkpoint_dir, exist_ok=True)
print(f'Checkpoints will be saved to: {drive_checkpoint_dir}')

# Train
!python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.005 \
    --max_size 1536 \
    --checkpoint_dir {drive_checkpoint_dir} \
    --device cuda
```

### Step 6: Resume Training (After Session Restart)

If your Colab session disconnects, you can resume:

```python
# Mount Drive again (for checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Clone repo again (dataset included automatically)
!git clone https://github.com/HarishSankarK/SGGF-Net.git
%cd SGGF-Net

# Dataset is already there - no need to copy!
!ls data/hit-uav/  # Verify

# Resume from checkpoint
drive_checkpoint_dir = '/content/drive/MyDrive/SGGF-Net-checkpoints'

!python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.005 \
    --max_size 1536 \
    --checkpoint_dir {drive_checkpoint_dir} \
    --resume {drive_checkpoint_dir}/latest.pth \
    --device cuda
```

### Step 7: Evaluate Model

```python
drive_checkpoint_dir = '/content/drive/MyDrive/SGGF-Net-checkpoints'

!python scripts/evaluate.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --checkpoint {drive_checkpoint_dir}/best.pth \
    --num_classes 6 \
    --split test
```

## Benefits of This Structure

1. **Organized**: Dataset in `data/` folder, easy to manage
2. **Persistent**: Checkpoints in Drive survive session restarts
3. **Resumable**: Can continue training from any checkpoint
4. **Clean**: Repository structure is consistent

## Tips

1. **First Time Setup**: Upload dataset to Drive once, then copy to `data/` each session
2. **Checkpoint Location**: Always use Drive path for `--checkpoint_dir` in Colab
3. **Resume**: Use `latest.pth` to continue from last epoch, or `best.pth` for best model
4. **Session Management**: Save important outputs to Drive before session ends

## Troubleshooting

- **Dataset not found**: Check path in Drive, verify copy command worked
- **Checkpoint not found**: Verify Drive path, check file exists
- **Out of memory**: Reduce batch_size to 1
- **Session timeout**: Use resume to continue training

