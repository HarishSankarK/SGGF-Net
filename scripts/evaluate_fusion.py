"""
Evaluation Script for Fusion-YOLOv11
Evaluates multimodal RGB-Thermal object detection model
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import FusionYOLOv11
from utils.dataset import DroneRGBTDataset, HITUAVDataset
from utils.transforms import get_val_transform
from utils.metrics import calculate_map, calculate_ap50, calculate_precision_recall_f1


def collate_fn_paired(batch):
    """Custom collate function for paired RGB-Thermal data"""
    rgb_images = [item[0][0] for item in batch]
    thermal_images = [item[0][1] for item in batch]
    targets = [item[1] for item in batch]
    return rgb_images, thermal_images, targets


def collate_fn_single(batch):
    """Custom collate function for single modality data"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def collate_fn_combined(batch):
    """
    Custom collate function for combined dataset (DroneRGBT + HIT-UAV)
    Handles both paired (RGB-Thermal) and single (RGB) data formats
    """
    rgb_images = []
    thermal_images = []
    targets = []
    
    for item in batch:
        # Check if it's paired data (DroneRGBT) or single data (HIT-UAV)
        if isinstance(item[0], tuple):
            # Paired RGB-Thermal data from DroneRGBT
            rgb_img, thermal_img = item[0]
            rgb_images.append(rgb_img)
            thermal_images.append(thermal_img)
        else:
            # Single modality (e.g. HIT-UAV) - duplicate for thermal
            rgb_img = item[0]
            rgb_images.append(rgb_img)
            thermal_images.append(rgb_img)  # Use same image for thermal
        
        targets.append(item[1])
    
    return rgb_images, thermal_images, targets


def evaluate(model, dataloader, device, num_classes, conf_threshold=0.5, nms_threshold=0.5):
    """
    Evaluate model on dataset
    
    Args:
        model: Fusion-YOLOv11 model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes (including background)
        conf_threshold: Confidence threshold for detections
        nms_threshold: NMS IoU threshold
    Returns:
        metrics: Dict with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc='Evaluating')
    
    with torch.no_grad():
        for batch_data in progress_bar:
            # Handle different dataset types
            # Paired data returns (rgb_images, thermal_images, targets) - 3 elements
            # Single modality returns (images, targets) - 2 elements
            if len(batch_data) == 3:
                # Paired RGB-Thermal (DroneRGBT)
                rgb_images, thermal_images, targets = batch_data
                rgb_images = [img.to(device) for img in rgb_images]
                thermal_images = [img.to(device) for img in thermal_images]
                
                # Get predictions with post-processing
                predictions = model.predict(rgb_images, thermal_images, 
                                           conf_threshold=conf_threshold,
                                           nms_threshold=nms_threshold)
            else:
                # Single modality
                images, targets = batch_data
                images = [img.to(device) for img in images]
                thermal_images = images
                
                # Get predictions with post-processing
                predictions = model.predict(images, thermal_images,
                                           conf_threshold=conf_threshold,
                                           nms_threshold=nms_threshold)
            
            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Compute metrics (use tensors on CPU — metrics expect torch tensors)
    formatted_targets = []
    for target in all_targets:
        formatted_targets.append({
            'boxes': target['boxes'].cpu(),
            'labels': target['labels'].cpu()
        })
    
    formatted_predictions = []
    for pred in all_predictions:
        formatted_predictions.append({
            'boxes': pred['boxes'].cpu(),
            'scores': pred['scores'].cpu(),
            'labels': pred['labels'].cpu()
        })
    
    # Compute mAP (requires num_classes parameter)
    map_50 = calculate_ap50(formatted_predictions, formatted_targets, num_classes)
    map_50_95 = calculate_map(formatted_predictions, formatted_targets, num_classes)
    
    # Compute precision, recall, F1
    precision, recall, f1 = calculate_precision_recall_f1(formatted_predictions, formatted_targets, num_classes)
    
    metrics = {
        'mAP@0.5': map_50,
        'mAP@0.5:0.95': map_50_95,
        'AP50': map_50,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print('\nEvaluation Results:')
    print(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"  AP50: {metrics['AP50']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fusion-YOLOv11')
    parser.add_argument('--dataset', type=str, default='dronergbt',
                       choices=['dronergbt', 'hituav', 'combined_rgbt_hituav'],
                       help='Dataset to evaluate on')
    parser.add_argument('--data_dir', type=str, 
                       default='sggf_net/data/dronergbt',
                       help='Dataset root directory (for single dataset)')
    parser.add_argument('--dronergbt_dir', type=str,
                       default='sggf_net/data/DroneRGBT',
                       help='DroneRGBT dataset directory (for combined evaluation)')
    parser.add_argument('--hituav_dir', type=str,
                       default='data/hit-uav',
                       help='HIT-UAV dataset directory (data/hit-uav or data/HIT-UAV)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (DroneRGBT: 2, HIT-UAV: 3)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='NMS IoU threshold')
    parser.add_argument('--max_size', type=int, default=640,
                       help='Max image size (default 640 for laptop/Colab GPUs)')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'],
                       help='Backbone (default: resnet50; overridden by checkpoint if saved)')
    parser.add_argument('--laptop', action='store_true',
                       help='Laptop mode: batch_size=2, max_size=640')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU (no GPU). Uses batch_size=1, num_workers=0, max_size=480 for speed.')
    
    args = parser.parse_args()
    
    if args.laptop:
        if '--batch_size' not in ' '.join(sys.argv):
            args.batch_size = 2
        if '--max_size' not in ' '.join(sys.argv):
            args.max_size = 640
    if args.cpu:
        if '--batch_size' not in ' '.join(sys.argv):
            args.batch_size = 1
        if '--max_size' not in ' '.join(sys.argv):
            args.max_size = 480
    
    # Device setup (--cpu forces CPU even if CUDA is available)
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using device: {device}')
    
    # Set num_classes based on dataset if not explicitly provided
    if args.num_classes == 2:  # Default value
        if args.dataset == 'dronergbt':
            args.num_classes = 2  # DroneRGBT: background + person
        elif args.dataset == 'combined_rgbt_hituav':
            args.num_classes = 3  # person + vehicle
        elif args.dataset == 'hituav':
            args.num_classes = 3  # HIT-UAV: person + vehicle
    
    # Load dataset
    val_transform = get_val_transform(max_size=args.max_size)
    
    if args.dataset == 'dronergbt':
        dataset = DroneRGBTDataset(
            root_dir=args.data_dir, split=args.split, transform=val_transform
        )
        collate_fn = collate_fn_paired
    elif args.dataset == 'hituav':
        dataset = HITUAVDataset(
            root_dir=args.data_dir, split=args.split, transform=val_transform, use_person_vehicle=True
        )
        collate_fn = collate_fn_single
    elif args.dataset == 'combined_rgbt_hituav':
        hituav_ds = HITUAVDataset(root_dir=args.hituav_dir, split=args.split, transform=val_transform, use_person_vehicle=True)
        dronergbt_ds = DroneRGBTDataset(root_dir=args.dronergbt_dir, split=args.split, transform=val_transform)
        dataset = ConcatDataset([hituav_ds, dronergbt_ds])
        collate_fn = collate_fn_combined
        print(f"✓ Combined loaded: {len(hituav_ds)} HIT-UAV + {len(dronergbt_ds)} DroneRGBT = {len(dataset)} total")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # CPU-friendly: no pin_memory, num_workers=0 to avoid multiprocessing overhead
    use_cuda = device.type == 'cuda'
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda
    )
    
    # Load checkpoint first to get backbone and num_classes (so model architecture matches)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone = checkpoint.get('backbone', args.backbone) if isinstance(checkpoint, dict) else args.backbone
    # Model num_classes = foreground classes (2 for person+vehicle). Metrics use args.num_classes (3) to iterate classes 1,2
    model_num_classes = checkpoint.get('num_classes') if isinstance(checkpoint, dict) else None
    if model_num_classes is None:
        model_num_classes = 2 if args.num_classes >= 3 else 1
    
    # Create model with same backbone and num_classes as checkpoint
    model = FusionYOLOv11(
        num_classes=model_num_classes,
        pretrained=False,
        fusion_type='concat_attention',
        backbone=backbone
    )
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("Loaded EMA weights from checkpoint")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model_state_dict from checkpoint")
    else:
        model.load_state_dict(checkpoint)
    if isinstance(checkpoint, dict):
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'best_map' in checkpoint:
            print(f"Best mAP in checkpoint: {checkpoint['best_map']:.4f}")
    
    model = model.to(device)
    model.eval()
    
    # Evaluate
    metrics = evaluate(
        model, dataloader, device, args.num_classes,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold
    )
    
    return metrics


if __name__ == '__main__':
    main()
