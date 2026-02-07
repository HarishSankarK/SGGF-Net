"""
Default configuration for Fusion-YOLOv11
NOTE: This config file is currently NOT used by training/evaluation scripts.
Scripts use command-line arguments instead. This file is kept for reference.
"""

# Dataset configuration
DATASET_CONFIG = {
    'dronergbt': {
        'num_classes': 2,  # background + person
        'max_size': 1536,
        'classes': [
            'background', 'person'
        ],
        'original_classes': ['person'],  # Only class present in actual dataset
        'data_format': 'paired',  # RGB-Thermal pairs
        'structure': {
            'rgb_dir': 'rgb',
            'thermal_dir': 'thermal',
            'annotations_dir': 'annotations'
        }
    },
    'smod': {
        'num_classes': 3,  # background + person + vehicle
        'max_size': 1536,
        'classes': [
            'background', 'person', 'vehicle'
        ],
        'original_classes': ['person', 'rider', 'bicycle', 'car'],
        'class_mapping': {
            'person': 0,  # person (0) and rider (1) → person (1)
            'vehicle': 1   # bicycle (2) and car (3) → vehicle (2)
        },
        'data_format': 'single',  # RGB only
        'structure': {
            'images_dir': 'images',
            'labels_dir': 'labels'
        }
    },
    'combined': {
        'num_classes': 3,  # background + person + vehicle
        'max_size': 1536,
        'classes': [
            'background', 'person', 'vehicle'
        ],
        'datasets': ['dronergbt', 'smod'],
        'data_format': 'mixed'  # Both paired and single
    },
    'hituav': {
        'num_classes': 6,  # 5 classes + background
        'max_size': 1536,
        'classes': [
            'background', 'Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare'
        ],
        'data_format': 'single'  # Thermal only
    }
}

# Model configuration (Fusion-YOLOv11)
MODEL_CONFIG = {
    'fusion_yolov11': {
        'num_classes': 3,  # Default for combined dataset
        'pretrained': True,
        'embed_dim': 192,  # GFEM embedding dimension
        'patch_size': 32,  # GFEM patch size
        'num_layers': 3,   # Number of transformer layers in GFEM
        'num_heads': 6,    # Number of attention heads in GFEM
        'fusion_type': 'concat_attention'  # Fusion strategy
    },
    'dual_stream': {
        'rgb_backbone': 'resnet50',
        'thermal_backbone': 'resnet50',
        'pretrained': True
    },
    'panet': {
        'in_channels_list': [256, 512, 1024, 2048],
        'out_channels': 256
    },
    'yolo_head': {
        'in_channels': 256,
        'num_anchors': 1  # Anchor-free
    }
}

# Training configuration (Fusion-YOLOv11)
TRAIN_CONFIG = {
    'batch_size': 4,  # Default, adjust based on GPU memory
    'num_epochs': 100,
    'lr': 1e-3,  # Learning rate
    'weight_decay': 5e-4,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
    'use_amp': False,  # Mixed precision (enable for CUDA)
    'checkpoint_dir': 'sggf_net/checkpoints',
    'save_best': True,
    'save_latest': True
}

# Evaluation configuration (Fusion-YOLOv11)
EVAL_CONFIG = {
    'batch_size': 4,
    'conf_threshold': 0.5,  # Confidence threshold for detections
    'nms_threshold': 0.5,   # NMS IoU threshold
    'iou_thresholds': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    'metrics': ['mAP@0.5', 'mAP@0.5:0.95', 'AP50', 'precision', 'recall', 'f1']
}

