# Dataset in Repository

## ✅ HIT-UAV Dataset Included

The HIT-UAV dataset is **included in this repository** at:
```
data/hit-uav/
├── images/
│   ├── train/    (2,008 images)
│   ├── val/      (287 images)
│   └── test/     (571 images)
└── labels/
    ├── train/    (2,008 labels)
    ├── val/      (287 labels)
    └── test/     (571 labels)
```

## 📊 Statistics

- **Total Size**: ~206MB
- **Total Files**: 5,733 files
- **Images**: 2,866 (train: 2,008, val: 287, test: 571)
- **Labels**: 2,867 (YOLO format .txt files)

## 🚀 Usage

### In Colab

```python
# Clone repository (dataset included!)
!git clone https://github.com/HarishSankarK/SGGF-Net.git
%cd SGGF-Net

# Dataset is already there!
!ls data/hit-uav/

# Start training immediately
!python scripts/train.py --dataset hituav --data_dir data/hit-uav
```

### Locally

```bash
# Dataset is already in the repository
cd sggf_net
python scripts/train.py --dataset hituav --data_dir data/hit-uav
```

## ⚠️ GitHub Limits

- **File size limit**: 100MB per file ✅ (our images are <100KB each)
- **Repository size**: Recommended <1GB ✅ (our repo is ~208MB)
- **Total files**: 5,733 files ✅ (well within limits)

## 📝 Notes

- Dataset is version-controlled with the code
- No need to upload to Drive or Colab
- Same dataset version for all users
- Easy to reproduce experiments

