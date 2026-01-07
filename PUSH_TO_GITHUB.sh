#!/bin/bash
# Script to push SGGF-Net to GitHub

cd /Users/harishshankar/Documents/Project/sggf_net

echo "Initializing Git repository..."
git init

echo "Adding all files..."
git add .

echo "Creating initial commit..."
git commit -m "Initial commit: SGGF-Net implementation with HIT-UAV dataset

- Complete SGGF-Net implementation (GFEM, NDPA, ARPM)
- HIT-UAV dataset included (206MB, 5,733 files) in data/hit-uav/
- Training and evaluation scripts
- Google Colab notebook with Drive checkpoint support
- Support for VisDrone, AI-TOD, and HIT-UAV datasets
- All code and dataset ready for Colab training"

echo "Adding remote repository..."
git remote add origin https://github.com/HarishSankarK/SGGF-Net.git

echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "Done! Repository pushed to https://github.com/HarishSankarK/SGGF-Net"

