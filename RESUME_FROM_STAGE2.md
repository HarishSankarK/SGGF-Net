# How to Resume from Stage 2 After Session Expiry

If your Colab session expired and you want to resume from Stage 2 with a different account:

## Step-by-Step Guide

### Option 1: Transfer Checkpoint Between Google Accounts (Recommended)

1. **From Previous Account:**
   - Open Google Drive from your previous account
   - Navigate to: `SGGF-Net-checkpoints/stage1_best.pth`
   - Download the checkpoint file (~800-900 MB)

2. **In New Colab Session:**
   - Upload the notebook: `SGGF_Net_Training_Colab.ipynb`
   - Run Cell 1 (Mount Drive and clone repository)
   - Run Cell 2 (Install dependencies)
   - **Run Cell 4** (Check Available Checkpoints) - This will show what's in your Drive
   - If `stage1_best.pth` is not found, upload it:
     - In Colab: Click **Files** → **Upload to session storage**
     - Upload `stage1_best.pth`
     - Run this in a new cell:
       ```python
       !mkdir -p /content/drive/MyDrive/SGGF-Net-checkpoints
       !cp /content/stage1_best.pth /content/drive/MyDrive/SGGF-Net-checkpoints/
       ```

3. **Resume Training:**
   - **Skip Cell 7** (Stage 1 - you already have the checkpoint)
   - **Run Cell 9** (Stage 2) - It will automatically find and use `stage1_best.pth`
   - Continue with Cell 11 (Stage 3) after Stage 2 completes

### Option 2: Use Google Drive Sharing

1. **From Previous Account:**
   - In Google Drive, right-click `stage1_best.pth`
   - Select "Share" → "Get link" → "Anyone with the link"
   - Copy the link

2. **In New Colab Session:**
   - Mount Drive (Cell 1)
   - Run this in a new cell to download directly:
     ```python
     import gdown
     # Replace FILE_ID with the ID from your Google Drive share link
     # Link format: https://drive.google.com/file/d/FILE_ID/view
     file_id = "YOUR_FILE_ID_HERE"
     output = "/content/drive/MyDrive/SGGF-Net-checkpoints/stage1_best.pth"
     !mkdir -p /content/drive/MyDrive/SGGF-Net-checkpoints
     gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
     ```

### Option 3: If Checkpoint is Already in New Account's Drive

If you've already uploaded the checkpoint to your new account's Drive:

1. Run Cell 1 (Mount Drive)
2. Run Cell 2 (Install dependencies)
3. Run Cell 4 (Check checkpoints) - This will verify the checkpoint exists
4. **Skip Cell 7** (Stage 1)
5. **Run Cell 9** (Stage 2) - Training will resume automatically

## Quick Reference

- **Cell 1**: Mount Drive + Clone Repository
- **Cell 2**: Install Dependencies
- **Cell 4**: Check Available Checkpoints ← **Run this first to see what you have!**
- **Cell 7**: Stage 1 Training ← **SKIP if you have stage1_best.pth**
- **Cell 9**: Stage 2 Training ← **Run this to resume from Stage 2**
- **Cell 11**: Stage 3 Training
- **Cell 13**: Evaluation

## Troubleshooting

**Q: Checkpoint not found even after uploading?**
- Make sure the checkpoint is in: `/content/drive/MyDrive/SGGF-Net-checkpoints/`
- Run Cell 4 to verify the exact path
- Check file permissions in Google Drive

**Q: Different checkpoint location?**
- Modify the `checkpoint_dir` variable in Cell 9
- Or update the path: `stage1_checkpoint = '/your/custom/path/stage1_best.pth'`

**Q: Want to use `stage1_latest.pth` instead of `stage1_best.pth`?**
- The code automatically uses `stage1_best.pth` if available
- Falls back to `stage1_latest.pth` if best is not found
- Both will work fine for resuming

## File Sizes (Approximate)

- `stage1_best.pth`: ~800-900 MB
- `stage2_best.pth`: ~800-900 MB  
- `stage3_best.pth`: ~800-900 MB

Make sure you have enough Drive storage space!

