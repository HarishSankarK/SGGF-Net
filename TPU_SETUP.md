# TPU v5e-1 Setup Guide

## Overview

This guide explains how to run SGGF-Net on Google Cloud TPU v5e-1. TPUs are specialized hardware for deep learning that can train models much faster than GPUs.

## Key Differences from GPU Training

1. **Uses `torch_xla`** instead of regular PyTorch CUDA operations
2. **Larger batch sizes** - TPUs can handle batch_size=8+ efficiently
3. **Parallel data loading** - Uses `MpDeviceLoader` for TPU cores
4. **Mark step** - Requires `xm.mark_step()` after optimizer updates
5. **No pin_memory** - TPUs don't use CUDA pin_memory

## Setup Steps

### 1. Create TPU VM

```bash
# Create TPU VM with v5e-1
gcloud compute tpus tpu-vm create sggf-net-tpu \
    --zone=us-central2-b \
    --accelerator-type=v5e-1 \
    --version=tpu-vm-tf-2.13.0
```

### 2. SSH into TPU VM

```bash
gcloud compute tpus tpu-vm ssh sggf-net-tpu --zone=us-central2-b
```

### 3. Install Dependencies

```bash
# Install PyTorch XLA
pip install torch torchvision
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Install other dependencies
pip install -r requirements.txt
```

### 4. Clone Repository

```bash
git clone https://github.com/HarishSankarK/SGGF-Net.git
cd SGGF-Net
```

### 5. Run Training

```bash
python scripts/train_tpu.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 0.0005 \
    --max_size 800 \
    --warmup_epochs 5 \
    --checkpoint_dir checkpoints \
    --val_freq 10 \
    --use_amp
```

## Colab TPU Setup

If using Google Colab with TPU:

1. **Enable TPU Runtime**:
   - Runtime → Change runtime type → TPU

2. **Install torch_xla**:
   ```python
   !pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
   ```

3. **Run the notebook** - The notebook has been updated for TPU

## TPU-Specific Optimizations

### Batch Size
- **GPU**: batch_size=1 (memory limited)
- **TPU**: batch_size=8+ (TPU can handle larger batches)

### Data Loading
- Uses `MpDeviceLoader` for parallel loading across TPU cores
- More workers (8) for better throughput
- `drop_last=True` for fixed batch sizes

### Memory
- TPU v5e-1 has 16GB HBM per core
- Can handle larger models and batch sizes
- No need for extreme memory optimizations

### Performance
- **Expected speedup**: 3-5× faster than GPU
- **Throughput**: ~2-3× more samples/second

## Code Changes for TPU

### 1. Device Selection
```python
import torch_xla.core.xla_model as xm
device = xm.xla_device()  # Instead of torch.device('cuda')
```

### 2. Data Loading
```python
from torch_xla.distributed.parallel_loader import MpDeviceLoader
train_loader = MpDeviceLoader(train_loader, device)
```

### 3. Mark Step
```python
# After optimizer step
xm.mark_step()  # Required for TPU
```

### 4. Model Movement
```python
model = xm.send_cpu_data_to_device(model, device)
```

### 5. Checkpoint Saving
```python
xm.save(checkpoint, path)  # Instead of torch.save
```

## Troubleshooting

### Issue: "torch_xla not found"
**Solution**: Install torch_xla:
```bash
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

### Issue: "TPU device not found"
**Solution**: Make sure TPU runtime is enabled in Colab, or TPU VM is running

### Issue: "Out of memory"
**Solution**: Reduce batch_size or max_size:
```bash
--batch_size 4  # Reduce from 8
--max_size 640  # Reduce from 800
```

### Issue: "Slow training"
**Solution**: 
- Increase batch_size (TPU benefits from larger batches)
- Use more workers: `num_workers=8`
- Ensure data is on fast storage (TPU storage or GCS)

## Performance Tips

1. **Use larger batch sizes** - TPUs are optimized for large batches
2. **Enable AMP** - Mixed precision training is faster on TPU
3. **Use persistent workers** - Reduces data loading overhead
4. **Preload data** - Copy dataset to TPU storage for faster access

## Expected Results

With TPU v5e-1:
- **Training time**: ~5-10 minutes per epoch (vs 10-15 min on GPU)
- **Throughput**: ~100-200 samples/second
- **Memory usage**: ~8-12GB per core
- **Speedup**: 3-5× faster than GPU

## Files Modified for TPU

1. `scripts/train_tpu.py` - New TPU-specific training script
2. `SGGF_Net_Training.ipynb` - Updated for TPU runtime
3. `requirements.txt` - Added torch_xla dependency

## Notes

- TPU training is compatible with all optimizations (NaN fixes, speed optimizations)
- Checkpoints are saved in the same format (compatible with GPU)
- Can resume training from GPU checkpoints on TPU (and vice versa)

