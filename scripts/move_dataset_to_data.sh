#!/bin/bash
# Script to move dataset to data/ folder (for local setup)

cd /Users/harishshankar/Documents/Project/sggf_net

# Create data directory if it doesn't exist
mkdir -p data

# Check if dataset exists in parent directory
if [ -d "../hit-uav" ]; then
    echo "Moving dataset from ../hit-uav to data/hit-uav..."
    cp -r ../hit-uav ./data/hit-uav
    echo "✓ Dataset moved to data/hit-uav/"
    echo ""
    echo "You can now train with:"
    echo "  python scripts/train.py --dataset hituav --data_dir data/hit-uav"
    echo "  Or just: python scripts/train.py --dataset hituav (uses default data/hit-uav)"
else
    echo "⚠ Dataset not found at ../hit-uav"
    echo "Please ensure the dataset is in the parent directory or update the path in this script"
fi

# Verify structure
echo ""
echo "Dataset structure:"
ls -la data/hit-uav/ 2>/dev/null || echo "Dataset not found in data/hit-uav/"

