# Quick Start Guide

## 🚀 Push to GitHub (One-Time Setup)

**Run these commands in the `sggf_net` directory:**

```bash
# Navigate to sggf_net directory
cd /Users/harishshankar/Documents/Project/sggf_net

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create commit
git commit -m "Initial commit: SGGF-Net implementation for HIT-UAV dataset"

# Add remote
git remote add origin https://github.com/HarishSankarK/SGGF-Net.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Or use the automated script:**
```bash
cd /Users/harishshankar/Documents/Project/sggf_net
./PUSH_TO_GITHUB.sh
```

## 📍 Directory Structure

```
Project/                    ← Parent directory (don't run git here)
├── hit-uav/               ← Dataset (stays local, not pushed)
├── sggf_net/              ← Code repository (run git here ✅)
│   ├── models/
│   ├── scripts/
│   ├── .gitignore
│   └── ...
└── venv/                  ← Virtual env (not pushed)
```

## ✅ Verification

After pushing, verify on GitHub:
- Go to: https://github.com/HarishSankarK/SGGF-Net
- You should see all code files, but NOT:
  - ❌ `hit-uav/` folder
  - ❌ `venv/` folder
  - ❌ `checkpoints/` folder
  - ❌ `__pycache__/` folders

## 🔄 Use in Colab

After pushing to GitHub, in Colab:

```python
!git clone https://github.com/HarishSankarK/SGGF-Net.git
%cd SGGF-Net
```

Then follow the Colab notebook: `SGGF_Net_Training.ipynb`

