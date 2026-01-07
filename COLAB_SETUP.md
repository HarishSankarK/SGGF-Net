# Google Colab Setup Guide for SGGF-Net

## Quick Start in Colab

### Step 1: Clone Repository

```python
# Clone the repository
!git clone https://github.com/HarishSankarK/SGGF-Net.git
%cd SGGF-Net
```

### Step 2: Install Dependencies

```python
# Install PyTorch with CUDA support (Colab has CUDA by default)
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install -r requirements.txt
```

### Step 3: Dataset (Already in Repository!)

The HIT-UAV dataset is **already included** in the GitHub repository at `data/hit-uav/`!

When you clone the repository, the dataset comes with it automatically. Just verify:

```python
# Dataset is already there - just verify!
!ls -la data/hit-uav/
print("✓ Dataset found in repository!")
```

**No upload needed!** The dataset is part of the GitHub repository.

**Alternative: If dataset is not in GitHub (fallback)**
```python
# Only use this if dataset is NOT in the repository
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data
!cp -r /content/drive/MyDrive/hit-uav ./data/hit-uav
```

### Step 4: Verify Dataset

```python
# Test dataset loading (uses data/hit-uav by default)
!python scripts/test_hituav_dataset.py
```

### Step 5: Start Training (Checkpoints to Drive)

```python
# Setup checkpoint directory in Google Drive
import os
drive_checkpoint_dir = '/content/drive/MyDrive/SGGF-Net-checkpoints'
os.makedirs(drive_checkpoint_dir, exist_ok=True)

# Train the model (checkpoints automatically saved to Drive)
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

### Step 6: Resume Training from Drive Checkpoint

```python
# Resume from latest checkpoint
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

## Complete Colab Notebook

See `SGGF_Net_Training.ipynb` for a complete notebook with all steps.

## Tips for Colab

1. **Runtime**: Use GPU runtime (Runtime → Change runtime type → GPU)
2. **Session Timeout**: Colab sessions timeout after ~12 hours of inactivity
3. **Save Progress**: Regularly save checkpoints to Google Drive
4. **Memory**: Colab has ~15GB RAM, adjust batch_size if needed
5. **Disk Space**: Colab has ~80GB disk, enough for dataset and checkpoints

## Troubleshooting

- **CUDA out of memory**: Reduce batch_size to 1
- **Dataset not found**: Check the path is correct
- **Import errors**: Make sure you're in the SGGF-Net directory

