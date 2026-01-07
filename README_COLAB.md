# Quick Colab Setup Guide

## 🚀 Super Quick Start

1. **Open Colab**: https://colab.research.google.com/
2. **Upload Notebook**: Upload `SGGF_Net_Training.ipynb`
3. **Run All Cells**: Runtime → Run All
4. **Done!** Training starts automatically

## 📋 What Happens Automatically

1. ✅ Mounts Google Drive
2. ✅ Clones your GitHub repo
3. ✅ Installs dependencies
4. ✅ Copies dataset from Drive to `data/hit-uav/`
5. ✅ Saves checkpoints to Drive (`/content/drive/MyDrive/SGGF-Net-checkpoints/`)
6. ✅ Starts training

## 📁 Dataset Setup (One-Time)

Before first training, upload your HIT-UAV dataset to Google Drive:

1. Zip your `hit-uav` folder
2. Upload to Google Drive at: `MyDrive/hit-uav/`
3. Unzip it there (or upload already unzipped)

**Drive Structure:**
```
MyDrive/
└── hit-uav/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

## 🔄 Resume Training

If Colab disconnects, just run **Step 7** in the notebook to resume from the last checkpoint saved in Drive.

## 💾 Checkpoint Location

All checkpoints are saved to:
```
/content/drive/MyDrive/SGGF-Net-checkpoints/
├── latest.pth  (updated every epoch)
└── best.pth    (best model based on mAP)
```

These persist even after Colab session ends!

## ⚙️ Customization

To change checkpoint location, modify this in Step 5:
```python
drive_checkpoint_dir = '/content/drive/MyDrive/YOUR-CUSTOM-PATH'
```

## 🐛 Troubleshooting

- **Dataset not found**: Check it's in `MyDrive/hit-uav/` in Drive
- **Out of memory**: Change `--batch_size 2` to `--batch_size 1` in Step 5
- **Checkpoint not found**: Make sure you've run Step 5 at least once

