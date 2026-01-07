# GitHub Setup Instructions

## ⚠️ IMPORTANT: Where to Run These Commands

**Run ALL commands in the `sggf_net` directory, NOT in the Project directory!**

```bash
# Navigate to the sggf_net directory first
cd /Users/harishshankar/Documents/Project/sggf_net

# Then run all git commands from here
```

**Why?** 
- The `sggf_net/` folder contains only the code (what we want to push)
- The `Project/` folder contains datasets and other files (too large for GitHub)
- The `.gitignore` in `sggf_net/` will exclude unnecessary files

## Step 1: Initialize Git Repository

```bash
cd /Users/harishshankar/Documents/Project/sggf_net
git init
```

## Step 2: Add All Files

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: SGGF-Net implementation for HIT-UAV dataset"
```

## Step 4: Add Remote Repository

```bash
git remote add origin https://github.com/HarishSankarK/SGGF-Net.git
```

## Step 5: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Alternative: If Repository Already Exists on GitHub

If the repository already has files (like README), you may need to pull first:

```bash
git pull origin main --allow-unrelated-histories
# Resolve any conflicts if needed
git push -u origin main
```

## Files Included

The repository includes:
- ✅ All model implementations (GFEM, NDPA, ARPM, SGGF-Net)
- ✅ Dataset loaders (VisDrone, AI-TOD, HIT-UAV)
- ✅ Training and evaluation scripts
- ✅ Configuration files
- ✅ Documentation (README, COLAB_SETUP)
- ✅ Colab notebook

## Files Excluded (via .gitignore)

- ❌ Virtual environment (venv/)
- ❌ Checkpoints (*.pth)
- ❌ Dataset files (too large)
- ❌ Python cache (__pycache__)
- ❌ IDE files (.vscode/, .idea/)

## After Pushing

Once pushed, you can clone in Colab:

```python
!git clone https://github.com/HarishSankarK/SGGF-Net.git
```

