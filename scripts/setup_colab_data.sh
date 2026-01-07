#!/bin/bash
# Script to setup dataset in data/ folder for Colab

# Create data directory
mkdir -p data

# Copy dataset from Drive to data/ folder
# Update the path if your dataset is in a different location
if [ -d "/content/drive/MyDrive/hit-uav" ]; then
    cp -r /content/drive/MyDrive/hit-uav ./data/hit-uav
    echo "✓ Dataset copied from Drive to data/hit-uav/"
elif [ -d "../hit-uav" ]; then
    cp -r ../hit-uav ./data/hit-uav
    echo "✓ Dataset copied from parent directory to data/hit-uav/"
else
    echo "⚠ Dataset not found. Please upload it first."
fi

# Verify structure
echo ""
echo "Dataset structure:"
ls -la data/hit-uav/ 2>/dev/null || echo "Dataset not found in data/hit-uav/"

