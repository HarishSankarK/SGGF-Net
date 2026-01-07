# Pushing to GitHub with Dataset

## ✅ Dataset Status

The HIT-UAV dataset (206MB, 5,733 files) is now in:
```
sggf_net/data/hit-uav/
```

## 📊 Repository Size

- **Dataset**: ~206MB
- **Code**: ~2MB  
- **Total**: ~208MB ✅ (Well within GitHub limits)

## 🚀 Push to GitHub

### Step 1: Navigate to sggf_net directory

```bash
cd /Users/harishshankar/Documents/Project/sggf_net
```

### Step 2: Initialize Git (if not already done)

```bash
git init
```

### Step 3: Add All Files (including dataset)

```bash
git add .
```

This will include:
- ✅ All code files
- ✅ Dataset in `data/hit-uav/`
- ✅ Documentation
- ✅ Colab notebook

### Step 4: Create Commit

```bash
git commit -m "Initial commit: SGGF-Net with HIT-UAV dataset

- Complete SGGF-Net implementation (GFEM, NDPA, ARPM)
- HIT-UAV dataset included (206MB, 5,733 files)
- Training and evaluation scripts
- Google Colab notebook with Drive checkpoint support
- Support for VisDrone, AI-TOD, and HIT-UAV datasets"
```

### Step 5: Add Remote and Push

```bash
git remote add origin https://github.com/HarishSankarK/SGGF-Net.git
git branch -M main
git push -u origin main
```

## ⏱️ Push Time Estimate

- **First push**: ~5-10 minutes (206MB dataset)
- **Subsequent pushes**: Much faster (only changes)

## ⚠️ Important Notes

1. **GitHub File Size Limit**: 100MB per file
   - Our dataset: Many small files ✅ (safe)

2. **Repository Size**: Recommended <1GB
   - Our repo: ~208MB ✅ (safe)

3. **If Push Fails**:
   - Check internet connection
   - Verify GitHub authentication
   - Try pushing in smaller commits if needed

## ✅ Verification

After pushing, verify on GitHub:
1. Go to: https://github.com/HarishSankarK/SGGF-Net
2. Check that `data/hit-uav/` folder exists
3. Verify file count matches (5,733 files)

## 🔄 Using in Colab

After pushing, in Colab:

```python
# Clone repository (dataset included!)
!git clone https://github.com/HarishSankarK/SGGF-Net.git
%cd SGGF-Net

# Dataset is already there!
!ls data/hit-uav/

# Start training immediately
!python scripts/train.py --dataset hituav --data_dir data/hit-uav
```

**No dataset upload needed!** Everything comes from GitHub.

