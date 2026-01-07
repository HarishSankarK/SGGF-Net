# Dataset in GitHub Repository

## ✅ Dataset Included

The HIT-UAV dataset (206MB) is included in this repository at:
```
sggf_net/data/hit-uav/
├── images/
│   ├── train/    (2,008 images)
│   ├── val/      (287 images)
│   └── test/     (571 images)
└── labels/
    ├── train/    (2,008 labels)
    ├── val/      (287 labels)
    └── test/     (571 labels)
```

## 📦 Repository Size

- Dataset: ~206MB
- Code: ~2MB
- **Total**: ~208MB (well within GitHub limits)

## 🚀 Benefits

1. **Easy Colab Setup**: Just clone the repo, dataset is included
2. **No Upload Needed**: No need to upload to Drive or Colab
3. **Version Control**: Dataset version is tracked with code
4. **Reproducibility**: Same dataset version for everyone

## ⚠️ Important Notes

1. **GitHub Limits**:
   - File size limit: 100MB per file
   - Recommended repo size: <1GB
   - Our dataset: 206MB total ✅ (safe)

2. **Cloning in Colab**:
   - Full clone includes dataset automatically
   - No additional steps needed

3. **Updating Dataset**:
   - If you update the dataset, commit and push normally
   - Git will track the changes

## 📝 Usage

**In Colab:**
```python
# Clone repository (dataset included)
!git clone https://github.com/HarishSankarK/SGGF-Net.git
%cd SGGF-Net

# Dataset is already there!
!ls data/hit-uav/

# Start training immediately
!python scripts/train.py --dataset hituav --data_dir data/hit-uav
```

**Locally:**
```bash
# Dataset is already in the repository
cd sggf_net
python scripts/train.py --dataset hituav --data_dir data/hit-uav
```

## 🔄 If You Need to Exclude Dataset Later

If the repository becomes too large, you can:

1. Use Git LFS (Large File Storage):
   ```bash
   git lfs install
   git lfs track "data/hit-uav/**"
   ```

2. Or exclude it and use Drive/other storage:
   - Update `.gitignore` to exclude `data/hit-uav/`
   - Update Colab notebook to download from Drive

But for now, 206MB is perfectly fine for GitHub!

