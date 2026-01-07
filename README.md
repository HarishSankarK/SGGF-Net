# SGGF-Net: UAV Image Object Detection

**UAV Image Object Detection based on Self-Attention Guidance and Global Feature Fusion**

This is an implementation of the SGGF-Net architecture for detecting objects in UAV (drone) images, as described in the paper:

> "UAV image object detection based on self-attention guidance and global feature fusion"  
> Jing Bai, Haiyang Hu, Xiaojing Liu, Shanna Zhuang, Zhengyou Wang  
> Image and Vision Computing 151 (2024) 105262

## Features

- **GFEM (Global Feature Extraction Module)**: Uses self-attention mechanism to capture long-range dependencies
- **NDPA (Normal Distribution-based Prior Assigner)**: Improves small object detection using KL divergence
- **ARPM (Attention-guided ROI Pooling Module)**: Optimizes multi-scale feature fusion
- Support for **VisDrone2021**, **AI-TOD**, and **HIT-UAV** datasets
- Comprehensive evaluation metrics: mAP, AP50, Precision, Recall, F1
- **Google Colab ready**: Includes notebook for easy training in Colab

## Project Structure

```
sggf_net/
├── models/              # Model architectures
│   ├── gfem.py         # Global Feature Extraction Module
│   ├── ndpa.py         # Normal Distribution-based Prior Assigner
│   ├── arpm.py         # Attention-guided ROI Pooling Module
│   └── sggf_net.py     # Main SGGF-Net architecture
├── utils/               # Utility functions
│   ├── dataset.py      # Dataset loaders
│   ├── transforms.py   # Data augmentation
│   └── metrics.py      # Evaluation metrics
├── scripts/            # Training and evaluation scripts
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation script
├── configs/            # Configuration files
│   └── default_config.py
├── checkpoints/        # Saved model checkpoints
├── results/            # Evaluation results
└── requirements.txt    # Python dependencies
```

## Quick Start

### Option 1: Google Colab (Recommended)

1. Open the Colab notebook: `SGGF_Net_Training.ipynb`
2. Run all cells - it will automatically clone the repo, install dependencies, and start training
3. See `COLAB_SETUP.md` for detailed instructions

### Option 2: Local Installation

1. **Clone or navigate to the project directory:**
```bash
git clone https://github.com/HarishSankarK/SGGF-Net.git
cd SGGF-Net
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install PyTorch (if not already installed):**
```bash
# For CUDA (GPU support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision
```

## Dataset Preparation

### VisDrone2021 Dataset

1. **Clone the repository** (at the Project level, not inside sggf_net):
```bash
cd /Users/harishshankar/Documents/Project
git clone https://github.com/VisDrone/VisDrone-Dataset.git
```

2. **Download the actual dataset files**:
   - The repository contains README and download links
   - Download **VisDrone-DET** dataset files:
     - trainset (1.44 GB): [BaiduYun](https://github.com/VisDrone/VisDrone-Dataset) | [GoogleDrive](https://github.com/VisDrone/VisDrone-Dataset)
     - valset (0.07 GB): [BaiduYun](https://github.com/VisDrone/VisDrone-Dataset) | [GoogleDrive](https://github.com/VisDrone/VisDrone-Dataset)
     - testset-dev (0.28 GB): [BaiduYun](https://github.com/VisDrone/VisDrone-Dataset) | [GoogleDrive](https://github.com/VisDrone/VisDrone-Dataset)

3. **Extract and organize the dataset**:
   - Extract the downloaded files
   - Use the organization script:
```bash
python scripts/organize_visdrone.py --download_dir /path/to/extracted/files --output_dir ../visdrone2021
```

   Or manually organize into this structure:
```
visdrone2021/          # Can be anywhere, e.g., /Users/harishshankar/Documents/Project/visdrone2021
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/
```

4. **Annotation format**: Each image has a corresponding `.txt` file with format:
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

### AI-TOD Dataset

1. Download the AI-TOD dataset from: https://github.com/jwwangchn/AI-TOD

2. Organize the dataset in the following structure:
```
aitod/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/
```

3. Annotation format: PASCAL VOC XML format

### HIT-UAV Dataset (Infrared Thermal)

**HIT-UAV** is a high-altitude infrared thermal dataset with 2,898 images, suitable for object detection in infrared/thermal imaging scenarios.

**Key Features:**
- Infrared thermal images (grayscale, converted to RGB automatically)
- High-altitude scenarios (schools, parking lots, roads, playgrounds)
- 10 object classes: person, car, bus, truck, van, motor, bicycle, tricycle, awning-tricycle
- Supports multiple annotation formats (JSON, XML, TXT)

1. **Download the HIT-UAV dataset**:
   - Repository: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
   - Paper: https://www.nature.com/articles/s41597-023-02066-6

2. **Clone the repository**:
```bash
cd /Users/harishshankar/Documents/Project
git clone https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset.git
```

3. **Organize the dataset** (structure may vary, the loader will auto-detect):
```
hituav/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/  (or labels/)
    ├── train/
    ├── val/
    └── test/
```

4. **Note**: HIT-UAV images are infrared thermal (grayscale), but the loader automatically converts them to RGB (3 channels) for compatibility with the model. The model expects RGB input.

**Differences from VisDrone:**
- **HIT-UAV**: Infrared thermal images (better for low-light/nighttime)
- **VisDrone**: Visual light RGB images (better for daytime)
- Both are suitable for UAV object detection, choose based on your application scenario

## Usage

### Training

Train on HIT-UAV dataset (infrared thermal):
```bash
# Local training (dataset in data/ folder)
python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.005 \
    --max_size 1536 \
    --checkpoint_dir checkpoints
```

Train on VisDrone2021 dataset:
```bash
python scripts/train.py \
    --dataset visdrone \
    --data_dir data/visdrone2021 \
    --num_classes 11 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.005 \
    --max_size 1536 \
    --checkpoint_dir checkpoints
```

Train on AI-TOD dataset:
```bash
python scripts/train.py \
    --dataset aitod \
    --data_dir data/aitod \
    --num_classes 9 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.005 \
    --max_size 800 \
    --checkpoint_dir checkpoints
```

**In Google Colab (with Drive checkpoints):**
```python
# Setup Drive checkpoint directory
drive_checkpoint_dir = '/content/drive/MyDrive/SGGF-Net-checkpoints'
import os
os.makedirs(drive_checkpoint_dir, exist_ok=True)

# Train (checkpoints saved to Drive)
!python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --num_classes 6 \
    --batch_size 2 \
    --num_epochs 50 \
    --lr 0.005 \
    --max_size 1536 \
    --checkpoint_dir {drive_checkpoint_dir} \
    --device cuda
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py \
    --dataset visdrone \
    --data_dir /path/to/visdrone2021 \
    --checkpoint checkpoints/best.pth \
    --num_classes 11 \
    --split test
```

### Resume Training

**Local:**
```bash
python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --resume checkpoints/latest.pth \
    --num_classes 6 \
    --checkpoint_dir checkpoints
```

**Colab (from Drive checkpoint):**
```python
drive_checkpoint_dir = '/content/drive/MyDrive/SGGF-Net-checkpoints'

!python scripts/train.py \
    --dataset hituav \
    --data_dir data/hit-uav \
    --resume {drive_checkpoint_dir}/latest.pth \
    --num_classes 6 \
    --checkpoint_dir {drive_checkpoint_dir} \
    --device cuda
```

## Model Architecture

### SGGF-Net Components

1. **GFEM (Global Feature Extraction Module)**
   - Divides input image into 4×4 patches
   - Uses multi-head self-attention to capture global dependencies
   - Outputs enhanced feature maps

2. **NDPA (Normal Distribution-based Prior Assigner)**
   - Models bounding boxes as 2D normal distributions
   - Uses KL divergence to match priors with ground truth
   - Improves small object detection accuracy

3. **ARPM (Attention-guided ROI Pooling Module)**
   - Fuses multi-scale features using 5×5 convolutions
   - Applies self-attention for feature enhancement
   - Optimizes ROI feature representation

## Training Configuration

Default training parameters (as per paper):
- **Optimizer**: SGD
- **Learning Rate**: 0.005
- **Momentum**: 0.9
- **Batch Size**: 2
- **Max Image Size**: 1536×1536 (VisDrone) or 800×800 (AI-TOD)
- **Data Augmentation**: Random horizontal flip (prob=0.5), resize, pad

## Evaluation Metrics

The model is evaluated using:
- **mAP**: Mean Average Precision (IoU thresholds: 0.5:0.95)
- **AP50**: Average Precision at IoU=0.5
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

## Expected Results

Based on the paper, SGGF-Net achieves:
- **VisDrone2021**: mAP=37.8, AP50=61.0, Precision=62.1, Recall=53.3, F1=57.4
- **AI-TOD**: mAP=24.2, AP50=57.2, APs=31.7 (for small objects)

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 4090 or similar)
- **RAM**: At least 16GB
- **Storage**: Sufficient space for dataset and checkpoints

## Troubleshooting

1. **Out of Memory**: Reduce batch size or max image size
2. **Slow Training**: Use GPU acceleration, reduce number of workers
3. **Import Errors**: Ensure all dependencies are installed correctly

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{bai2024uav,
  title={UAV image object detection based on self-attention guidance and global feature fusion},
  author={Bai, Jing and Hu, Haiyang and Liu, Xiaojing and Zhuang, Shanna and Wang, Zhengyou},
  journal={Image and Vision Computing},
  volume={151},
  pages={105262},
  year={2024}
}
```

## License

This implementation is for educational and research purposes. Please refer to the original paper for licensing information.

## Contact

For questions or issues, please refer to the original paper authors or create an issue in the repository.

