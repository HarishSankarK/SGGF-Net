# Quick Start: Google Colab Training (RECOMMENDED)

**Problem**: CPU training is extremely slow (8+ minutes per batch) due to:
- NDPA matrix inverse/determinant operations (100k+ anchors)
- PyTorch compilation overhead
- CPU being ~10-20x slower than GPU

**Solution**: Use Google Colab with T4 GPU (FREE)

## Step-by-Step Guide

### 1. Open Colab Notebook

1. Go to: https://colab.research.google.com/
2. Upload `SGGF_Net_Training_Colab.ipynb` from your repo
   OR
   Clone repo directly in Colab

### 2. Enable GPU

- **Runtime** → **Change runtime type**
- Select: **GPU (T4)** 
- Click: **Save**

### 3. Run All Cells

Just click "Run All" or run cells sequentially. The notebook will:
- Mount Google Drive (for checkpoints)
- Clone repository from GitHub
- Install PyTorch with CUDA
- Install dependencies
- Run Stage 1, 2, 3 training
- Save checkpoints to Drive

### 4. Training Time

- **Stage 1**: ~15-20 minutes
- **Stage 2**: ~12-15 minutes  
- **Stage 3**: ~8-10 minutes
- **Total**: ~35-45 minutes

### 5. Checkpoints Location

All checkpoints saved to:
```
/content/drive/MyDrive/SGGF-Net-checkpoints/
  - stage1_best.pth
  - stage1_latest.pth
  - stage2_best.pth
  - stage2_latest.pth
  - stage3_best.pth
  - stage3_latest.pth
```

These persist even after Colab session ends!

## Why Colab is Better

| Feature | Local M1 CPU | Colab T4 GPU |
|---------|-------------|--------------|
| Speed | ~2-3 hours | ~35-45 min |
| Stability | Stable | Stable |
| Cost | Free | Free |
| First batch | 5-10 min | 10-20 sec |
| NDPA computation | Very slow | Fast |

## Troubleshooting

**Q: Colab says "GPU not available"**
- Check Runtime → Change runtime type → GPU (T4) is selected
- Free Colab users get T4, sometimes need to wait for availability

**Q: How to download checkpoints?**
- Checkpoints are in Google Drive
- Download from: https://drive.google.com/drive/my-drive
- Look in: `SGGF-Net-checkpoints/` folder

**Q: Can I resume training?**
- Yes! Checkpoints are saved after each stage
- Use `--resume` flag with the checkpoint path

## Alternative: Local Training (CPU - Slow but Works)

If you must use local CPU:
1. Be patient - first batch takes 5-10 minutes
2. Each batch after: ~30-60 seconds
3. Total time: 2-3 hours
4. Progress will show after first batch completes

The code now has verbose logging so you'll see progress messages.

