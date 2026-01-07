"""
Default configuration for SGGF-Net
"""

# Dataset configuration
DATASET_CONFIG = {
    'visdrone': {
        'num_classes': 11,  # 10 classes + background
        'max_size': 1536,
        'classes': [
            'ignored', 'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
        ]
    },
    'aitod': {
        'num_classes': 9,  # 8 classes + background
        'max_size': 800,
        'classes': [
            'background', 'airplane', 'bridge', 'oil-tank', 'boat',
            'swimming-pool', 'vehicle', 'person', 'windmill'
        ]
    },
    'hituav': {
        'num_classes': 6,  # 5 classes + background (Person, Car, Bicycle, OtherVehicle, DontCare)
        'max_size': 1536,
        'classes': [
            'background', 'Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare'
        ]
    }
}

# Model configuration
MODEL_CONFIG = {
    'gfem': {
        'in_channels': 3,
        'embed_dim': 256,
        'patch_size': 4,
        'num_layers': 4,
        'num_heads': 8,
        'mlp_ratio': 4,
        'dropout': 0.1
    },
    'ndpa': {
        'pos_threshold': 0.5,
        'neg_threshold': 0.3
    },
    'arpm': {
        'in_channels': 256,
        'out_channels': 256,
        'roi_size': 7
    }
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 2,
    'num_epochs': 50,
    'lr': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'lr_step_size': 10,
    'lr_gamma': 0.1,
    'max_size': 1536,
    'flip_prob': 0.5
}

# Evaluation configuration
EVAL_CONFIG = {
    'batch_size': 2,
    'iou_thresholds': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
}

