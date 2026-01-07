#!/bin/bash
# Script to start training on HIT-UAV dataset

cd /Users/harishshankar/Documents/Project/sggf_net

# Activate virtual environment
source ../venv/bin/activate

# Test dataset first
echo "Testing dataset loader..."
python scripts/test_hituav_dataset.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Starting training..."
    echo "=" * 50
    
    # Start training
    python scripts/train.py \
        --dataset hituav \
        --data_dir ../hit-uav \
        --num_classes 6 \
        --batch_size 2 \
        --num_epochs 50 \
        --lr 0.005 \
        --momentum 0.9 \
        --weight_decay 0.0001 \
        --max_size 1536 \
        --checkpoint_dir checkpoints \
        --device cuda
else
    echo "Dataset test failed. Please check the dataset path and structure."
    exit 1
fi

