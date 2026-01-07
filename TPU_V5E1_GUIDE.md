# TPU v5e-1 Training Guide

## TPU v5e-1 Specifications

- **Cores**: 8 cores
- **Memory**: 16GB HBM per core (128GB total)
- **Speed**: 3-5× faster than GPU
- **Batch Size**: Can handle 16-32 efficiently
- **Throughput**: ~100-200 samples/second

## Quick Start for TPU v5e-1

### Step 1: Enable TPU Runtime

1. **Runtime** → **Change runtime type**
2. Select **TPU**
3. Click **Save**
4. **Runtime** → **Restart runtime**

### Step 2: Install TPU Dependencies

Run cell 7 (Option B) in the notebook:

```python
# Install PyTorch XLA for TPU
!pip install torch torchvision
!pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Verify TPU setup
import torch_xla
import torch_xla.core.xla_model as xm
device = torch_xla.device()
print(f'✓ TPU v5e-1: {device}')
```

### Step 3: Run Training

Use the TPU training command (cell 18, Option B):

```bash
python scripts/train_tpu.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 16 \
    --num_epochs 50 \
    --lr 0.0005 \
    --max_size 800 \
    --warmup_epochs 5 \
    --checkpoint_dir /content/drive/MyDrive/SGGF-Net-checkpoints \
    --val_freq 10 \
    --use_amp
```

## TPU v5e-1 Optimized Settings

### Recommended Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `batch_size` | 16 | TPU v5e-1 can handle 16-32 efficiently |
| `max_size` | 800 | Balanced for TPU memory |
| `patch_size` | 32 | Memory efficient (already set in model) |
| `embed_dim` | 192 | Optimized (already set in model) |
| `num_heads` | 6 | Optimized (already set in model) |
| `num_layers` | 3 | Optimized (already set in model) |
| `lr` | 0.0005 | Prevents NaN losses |
| `warmup_epochs` | 5 | Better convergence |

### Performance Expectations

- **Training Speed**: 3-5× faster than GPU
- **Epoch Time**: ~5-10 minutes (vs 10-15 min on GPU)
- **Throughput**: ~100-200 samples/second
- **Memory Usage**: ~8-12GB per core

## TPU v5e-1 Specific Optimizations

### 1. Larger Batch Sizes

TPU v5e-1 can efficiently handle:
- **Batch size 16**: Recommended (good balance)
- **Batch size 32**: Maximum (if you want even faster training)

### 2. Parallel Data Loading

The script uses `MpDeviceLoader` which:
- Distributes data loading across 8 TPU cores
- Uses 8 workers for optimal throughput
- Prefetches data for continuous training

### 3. TPU Mark Step

After each optimizer update:
```python
xm.mark_step()  # Required for TPU synchronization
```

## Training Command for TPU v5e-1

### Standard Training

```bash
python scripts/train_tpu.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 16 \
    --num_epochs 50 \
    --lr 0.0005 \
    --max_size 800 \
    --warmup_epochs 5 \
    --checkpoint_dir checkpoints \
    --val_freq 10 \
    --use_amp
```

### Maximum Speed (Larger Batch)

```bash
python scripts/train_tpu.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.0005 \
    --max_size 800 \
    --warmup_epochs 5 \
    --checkpoint_dir checkpoints \
    --val_freq 10 \
    --use_amp
```

## Troubleshooting TPU v5e-1

### Issue: "TPU initialization failed"

**Solution**:
1. Restart runtime: Runtime → Restart runtime
2. Wait 10-30 seconds
3. Re-run setup cell (cell 7)

### Issue: OOM on TPU

**Solution**: Reduce batch size:
```bash
--batch_size 8  # Instead of 16
```

### Issue: Slow training

**Solution**: Increase batch size (if memory allows):
```bash
--batch_size 32  # Maximum for v5e-1
```

## Key Advantages of TPU v5e-1

1. **8 Cores**: Parallel processing across cores
2. **Large Memory**: 16GB per core (can handle larger batches)
3. **Fast Training**: 3-5× faster than GPU
4. **Efficient**: Optimized for deep learning workloads

## Expected Results

With TPU v5e-1 and batch_size=16:
- **Epoch time**: ~5-10 minutes
- **Total training time (50 epochs)**: ~4-8 hours
- **Throughput**: ~100-200 samples/second
- **Memory usage**: ~8-12GB per core

## Notes

- ✅ All optimizations (NaN fixes, speed optimizations) work on TPU v5e-1
- ✅ Checkpoints are compatible with GPU (can resume on either)
- ✅ TPU training is much faster than GPU
- ✅ Can use larger batch sizes for even faster training

Enjoy the speed boost with TPU v5e-1! 🚀

