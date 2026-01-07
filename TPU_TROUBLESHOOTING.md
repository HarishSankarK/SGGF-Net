# TPU Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: "TPU initialization failed: Device or resource busy"

**Cause**: TPU runtime is not properly enabled in Colab, or TPU is already in use.

**Solutions**:

1. **Enable TPU Runtime in Colab**:
   - Go to: Runtime → Change runtime type
   - Select: **TPU** (not GPU or CPU)
   - Click Save
   - Restart runtime

2. **Check TPU Status**:
   ```python
   import os
   print("TPU Runtime:", os.environ.get('COLAB_TPU_ADDR', 'Not set'))
   ```

3. **Restart Runtime**:
   - Runtime → Restart runtime
   - Re-run all cells from the beginning

### Issue 2: "AttributeError: module 'torch_xla.core.xla_model' has no attribute 'xla_world_size'"

**Cause**: Newer versions of torch_xla (2.9+) have different APIs.

**Solution**: The code has been updated to handle both old and new APIs. If you still see this error, use:

```python
# Instead of xm.xla_world_size()
try:
    world_size = torch_xla._XLAC._xla_get_replication_devices_count()
except:
    # Just skip if unavailable
    pass
```

### Issue 3: "DeprecationWarning: Use torch_xla.device instead"

**Cause**: `xm.xla_device()` is deprecated in torch_xla 2.9+.

**Solution**: Use the new API:
```python
import torch_xla
device = torch_xla.device()  # Instead of xm.xla_device()
```

The code has been updated to use the new API with fallback.

### Issue 4: TPU Not Found / Falls Back to CPU

**Solutions**:

1. **Verify TPU Runtime**:
   ```python
   import torch_xla
   try:
       device = torch_xla.device()
       print(f"TPU device: {device}")
   except RuntimeError as e:
       print(f"TPU not available: {e}")
   ```

2. **Check Colab TPU Availability**:
   - TPU runtime might not be available in free Colab
   - You may need Colab Pro or Colab Pro+
   - Or use Google Cloud TPU VM instead

3. **Alternative: Use GPU**:
   - If TPU is not available, the code will automatically fall back to GPU/CPU
   - Change runtime type to GPU
   - Use `scripts/train.py` instead of `scripts/train_tpu.py`

### Issue 5: "Couldn't open iommu group /dev/vfio/0"

**Cause**: TPU device is busy or not properly initialized.

**Solutions**:

1. **Restart Runtime**:
   - Runtime → Restart runtime
   - Re-run setup cells

2. **Check for Other Processes**:
   - Make sure no other TPU processes are running
   - Close other notebooks using TPU

3. **Wait and Retry**:
   - Sometimes TPU needs a moment to initialize
   - Wait 10-30 seconds and retry

### Issue 6: Slow Training on TPU

**Solutions**:

1. **Increase Batch Size**:
   - TPUs work best with larger batches
   - Try `--batch_size 16` or `--batch_size 32`

2. **Use More Workers**:
   - Increase `num_workers` in DataLoader
   - TPU can handle 8-16 workers

3. **Preload Data**:
   - Copy dataset to TPU storage for faster access
   - Use persistent workers

## Fallback to GPU/CPU

If TPU is not available, the code will automatically fall back:

1. **For Colab GPU**:
   ```python
   # Use regular training script
   !python scripts/train.py \
       --dataset hituav \
       --data_dir data/hit-uav \
       --num_classes 6 \
       --batch_size 1 \
       --num_epochs 50 \
       --lr 0.0005 \
       --max_size 800 \
       --checkpoint_dir {drive_checkpoint_dir} \
       --device cuda \
       --val_freq 10 \
       --use_amp
   ```

2. **The train_tpu.py script will automatically detect and fall back**:
   - If TPU initialization fails, it uses GPU/CPU
   - All optimizations still work
   - Just slower than TPU

## Verification Steps

1. **Check TPU Runtime**:
   ```python
   import os
   print("Runtime type:", os.environ.get('COLAB_TPU_ADDR', 'Not TPU'))
   ```

2. **Test TPU Device**:
   ```python
   import torch_xla
   try:
       device = torch_xla.device()
       print(f"✓ TPU device: {device}")
       # Test a simple operation
       x = torch.randn(2, 3).to(device)
       print(f"✓ TPU tensor created: {x.device}")
   except Exception as e:
       print(f"✗ TPU not available: {e}")
   ```

3. **Check torch_xla Version**:
   ```python
   import torch_xla
   print(f"torch_xla version: {torch_xla.__version__}")
   ```

## Recommended Setup

1. **Colab Pro/Pro+** (for TPU access)
2. **Or Google Cloud TPU VM** (for dedicated TPU)
3. **Or Colab GPU** (free, but slower)

The code works with all three options!

