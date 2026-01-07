# Training Instructions for HIT-UAV Dataset

## Quick Start

Your HIT-UAV dataset is located at: `/Users/harishshankar/Documents/Project/hit-uav/`

### Step 1: Test Dataset Loading

First, verify the dataset loads correctly:

```bash
cd /Users/harishshankar/Documents/Project/sggf_net
source ../venv/bin/activate
python scripts/test_hituav_dataset.py
```

### Step 2: Start Training

**Option A: Using the training script directly**
```bash
cd /Users/harishshankar/Documents/Project/sggf_net
source ../venv/bin/activate
python scripts/train.py \
    --dataset hituav \
    --data_dir ../hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.005 \
    --max_size 1536 \
    --checkpoint_dir checkpoints
```

**Option B: Using the shell script**
```bash
cd /Users/harishshankar/Documents/Project/sggf_net
./scripts/start_training.sh
```

## Dataset Configuration

- **Dataset**: HIT-UAV (Infrared Thermal)
- **Location**: `/Users/harishshankar/Documents/Project/hit-uav/`
- **Classes**: 6 (background + Person, Car, Bicycle, OtherVehicle, DontCare)
- **Training images**: 2,008
- **Validation images**: 287
- **Test images**: 571

## Training Parameters

- **Batch Size**: 2 (as per paper, due to large image sizes)
- **Learning Rate**: 0.005
- **Momentum**: 0.9
- **Epochs**: 50
- **Max Image Size**: 1536×1536
- **Optimizer**: SGD
- **LR Schedule**: Step decay (factor 0.1 every 10 epochs)

## Expected Training Time

- **With GPU (CUDA)**: ~2-4 hours for 50 epochs
- **With CPU**: Much longer (not recommended)

## Monitoring Training

Training progress will show:
- Epoch number
- Batch progress
- Loss values
- Validation metrics (mAP, AP50, Precision, Recall, F1)

## Checkpoints

Checkpoints are saved in `sggf_net/checkpoints/`:
- `latest.pth` - Latest checkpoint (every epoch)
- `best.pth` - Best model based on mAP

## Resume Training

If training is interrupted, resume from checkpoint:
```bash
python scripts/train.py \
    --dataset hituav \
    --data_dir ../hit-uav \
    --num_classes 6 \
    --resume checkpoints/latest.pth \
    --checkpoint_dir checkpoints
```

## Evaluation

After training, evaluate on test set:
```bash
python scripts/evaluate.py \
    --dataset hituav \
    --data_dir ../hit-uav \
    --checkpoint checkpoints/best.pth \
    --num_classes 6 \
    --split test
```

## Troubleshooting

1. **Out of Memory**: Reduce batch_size to 1 or reduce max_size
2. **CUDA not available**: Training will use CPU (very slow)
3. **Dataset not found**: Check the path `../hit-uav` is correct
4. **Import errors**: Make sure virtual environment is activated

## Notes

- The HIT-UAV dataset uses YOLO format labels (normalized coordinates)
- Images are automatically converted from grayscale to RGB
- DontCare class (class 4) is filtered out during training
- The model expects 3-channel RGB input

