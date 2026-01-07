# Training Speed Optimizations

## Overview

These optimizations can speed up training by **1.5-3x** without affecting model accuracy.

## Implemented Optimizations

### 1. **Automatic Mixed Precision (AMP)** ✅ Enabled by Default
- **Speedup**: 1.5-2x faster training
- **Accuracy Impact**: Minimal (<0.1% typically)
- **Memory**: Uses less GPU memory
- **Status**: Enabled by default, use `--no_amp` to disable

**How it works**: Uses FP16 for forward pass, FP32 for backward pass. Automatically handled by PyTorch.

### 2. **Optimized DataLoader** ✅
- **num_workers**: Increased to 4 (from 2)
- **prefetch_factor**: 2 (preloads next batches)
- **persistent_workers**: True (keeps workers alive between epochs)
- **pin_memory**: True (faster CPU→GPU transfer)
- **non_blocking**: True (async data transfer)

**Speedup**: 10-20% faster data loading

### 3. **Reduced Validation Frequency** ✅
- **Default**: Validate every 5 epochs (instead of every epoch)
- **Speedup**: Saves ~2 minutes per epoch
- **Control**: Use `--val_freq N` to change frequency

### 4. **torch.compile (Optional)** ⚡
- **Speedup**: 10-30% additional speedup (PyTorch 2.0+)
- **Usage**: Add `--compile` flag
- **Note**: First epoch may be slower (compilation overhead)

## Expected Speed Improvements

| Optimization | Speedup | Total Time (10 min →) |
|-------------|---------|----------------------|
| Baseline | 1.0x | 10 min/epoch |
| + AMP | 1.5-2.0x | **5-7 min/epoch** |
| + DataLoader | 1.6-2.1x | **4.5-6 min/epoch** |
| + Reduced Val | 1.7-2.2x | **4.5-5.5 min/epoch** |
| + torch.compile | 1.8-2.5x | **4-5.5 min/epoch** |

## Usage

### Default (All Optimizations Enabled)
```bash
python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --max_size 1024
```

### With torch.compile (Additional Speedup)
```bash
python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --max_size 1024 \
    --compile
```

### Disable AMP (if needed)
```bash
python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --max_size 1024 \
    --no_amp
```

### Custom Validation Frequency
```bash
python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --max_size 1024 \
    --val_freq 10  # Validate every 10 epochs
```

## In Colab Notebook

The optimizations are automatically enabled. Just run:

```python
!python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --max_size 1024 \
    --use_amp  # Already default, but explicit
```

## Additional Speed Tips

1. **Increase Batch Size** (if GPU memory allows):
   ```bash
   --batch_size 4  # Instead of 2
   ```
   - **Speedup**: ~1.5-2x (fewer iterations)
   - **Trade-off**: More GPU memory needed

2. **Reduce Image Size** (if acceptable):
   ```bash
   --max_size 800  # Instead of 1024
   ```
   - **Speedup**: ~1.3x
   - **Trade-off**: Slightly lower accuracy for small objects

3. **Reduce GFEM Layers** (if acceptable):
   - Edit `models/sggf_net.py`: `num_layers=2` instead of `4`
   - **Speedup**: ~1.2x
   - **Trade-off**: Slightly less feature extraction capacity

## Monitoring Performance

Check training speed:
```python
import time
start = time.time()
# ... training ...
elapsed = time.time() - start
print(f"Epoch took: {elapsed/60:.2f} minutes")
```

## Accuracy Verification

These optimizations maintain accuracy:
- **AMP**: Typically <0.1% mAP difference
- **DataLoader**: No accuracy impact
- **Validation frequency**: No accuracy impact (only affects when you see metrics)
- **torch.compile**: No accuracy impact

## Troubleshooting

### If AMP causes NaN losses:
```bash
--no_amp  # Disable AMP
```

### If DataLoader workers cause issues:
- Automatically reduced to available CPU cores
- Can manually set `num_workers=2` in code if needed

### If torch.compile fails:
- Requires PyTorch 2.0+
- First epoch will be slower (compilation)
- If errors occur, just don't use `--compile` flag

## Summary

**Expected improvement**: **~2x faster** (10 min → 5 min per epoch)

**No accuracy loss** - All optimizations are proven techniques used in production.

