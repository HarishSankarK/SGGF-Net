"""
Training Script for Fusion-YOLOv11
Dual-stream RGB-Thermal multimodal object detection
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import FusionYOLOv11
from utils.dataset import DroneRGBTDataset, HITUAVDataset, SMODDataset
from utils.transforms import get_train_transform, get_val_transform
from utils.metrics import calculate_map, calculate_ap50


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
    Custom collate function for combined dataset (DroneRGBT + SMOD)
    Handles both paired (RGB-Thermal) and single (RGB) data formats
    """
    rgb_images = []
    thermal_images = []
    targets = []
    
    for item in batch:
        # Check if it's paired data (DroneRGBT) or single data (SMOD)
        if isinstance(item[0], tuple):
            # Paired RGB-Thermal data from DroneRGBT
            rgb_img, thermal_img = item[0]
            rgb_images.append(rgb_img)
            thermal_images.append(thermal_img)
        else:
            # Single RGB data from SMOD - duplicate for thermal
            rgb_img = item[0]
            rgb_images.append(rgb_img)
            thermal_images.append(rgb_img)  # Use same image for thermal
        
        targets.append(item[1])
    
    return rgb_images, thermal_images, targets


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, use_amp=False, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Reduce progress bar update frequency for better performance
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]', mininterval=1.0)
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # Handle different dataset types
        # Paired data returns (rgb_images, thermal_images, targets) - 3 elements
        # Single modality returns (images, targets) - 2 elements
        if len(batch_data) == 3:
            # Paired RGB-Thermal (DroneRGBT)
            rgb_images, thermal_images, targets = batch_data
            # Use non_blocking transfer for faster GPU transfer (overlaps with computation)
            rgb_images = [img.to(device, non_blocking=True) for img in rgb_images]
            thermal_images = [img.to(device, non_blocking=True) for img in thermal_images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss_dict = model(rgb_images, thermal_images, targets)
                    loss = sum(loss for loss in loss_dict.values())
            else:
                loss_dict = model(rgb_images, thermal_images, targets)
                loss = sum(loss for loss in loss_dict.values())
        else:
            # Single modality - create dummy thermal images for now
            images, targets = batch_data
            # Use non_blocking transfer for faster GPU transfer
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # For single modality, use same image for both RGB and Thermal
            # (This allows training on HIT-UAV or SMOD alone)
            thermal_images = images
            
            optimizer.zero_grad()
            
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss_dict = model(images, thermal_images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    # Check for NaN in individual loss components
                    if any(torch.isnan(l) or torch.isinf(l) for l in loss_dict.values()):
                        print(f'\n⚠️  Warning: NaN/Inf in loss components at batch {batch_idx}!')
                        for k, v in loss_dict.items():
                            if torch.isnan(v) or torch.isinf(v):
                                print(f'  {k}: {v.item()}')
            else:
                loss_dict = model(images, thermal_images, targets)
                loss = sum(loss for loss in loss_dict.values())
                # Check for NaN in individual loss components
                if any(torch.isnan(l) or torch.isinf(l) for l in loss_dict.values()):
                    print(f'\n⚠️  Warning: NaN/Inf in loss components at batch {batch_idx}!')
                    for k, v in loss_dict.items():
                        if torch.isnan(v) or torch.isinf(v):
                            print(f'  {k}: {v.item()}')
        
        # Check for NaN/Inf loss before backward pass (applies to both paired and single modality)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f'\n⚠️  Warning: NaN/Inf loss detected at batch {batch_idx}! Skipping batch...')
            optimizer.zero_grad()
            continue
        
        if use_amp and device.type == 'cuda':
            scaler.scale(loss).backward()
            # Gradient clipping before optimizer step
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping before optimizer step
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar (less frequently for better performance)
        if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def freeze_backbone_layers(model, freeze_early=True, freeze_layers=['layer0', 'layer1', 'layer2']):
    """
    Freeze ResNet backbone layers for transfer learning
    
    Args:
        model: FusionYOLOv11 model
        freeze_early: If True, freeze early layers (layer0, layer1, layer2)
        freeze_layers: List of layer names to freeze
    """
    frozen_count = 0
    total_count = 0
    
    # Freeze RGB backbone layers
    for name, param in model.dual_stream.rgb_backbone.named_parameters():
        total_count += 1
        if freeze_early:
            # Freeze layer0, layer1, layer2 (early feature extraction)
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
        else:
            param.requires_grad = True
    
    # Freeze Thermal backbone layers (same structure)
    for name, param in model.dual_stream.thermal_backbone.named_parameters():
        if freeze_early:
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    if freeze_early:
        print(f'✓ Transfer Learning: Frozen {frozen_count}/{total_count} early backbone layers')
        print(f'  Frozen: {", ".join(freeze_layers)}')
        print(f'  Training: layer3, layer4, GFEM, Fusion, PANet, Detection Head')
    else:
        print('✓ Training all layers (no freezing)')
    
    return frozen_count


def unfreeze_all_layers(model):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True
    print('✓ Unfrozen all layers')


def freeze_for_stage(model, stage):
    """
    Stage-by-stage training for FusionYOLOv11 (speeds up training).
    
    Stage 1: Train backbone (layer3, layer4) + fusion + PANet + head
             Frozen: GFEM (transformer), backbone layer0/1/2
    Stage 2: Train only GFEM modules (freeze everything else)
    Stage 3: Full fine-tuning (unfreeze all, lower LR)
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    if stage == 1:
        # Unfreeze: backbone layer3, layer4 + fusion_conv + fusion modules + panet + detection_head
        freeze_layers = ['layer0', 'layer1', 'layer2']
        for name, param in model.dual_stream.rgb_backbone.named_parameters():
            if not any(l in name for l in freeze_layers):
                param.requires_grad = True
        for name, param in model.dual_stream.thermal_backbone.named_parameters():
            if not any(l in name for l in freeze_layers):
                param.requires_grad = True
        for param in model.dual_stream.rgb_fusion_conv.parameters():
            param.requires_grad = True
        for param in model.dual_stream.thermal_fusion_conv.parameters():
            param.requires_grad = True
        for m in [model.fusion_c2, model.fusion_c3, model.fusion_c4, model.fusion_c5]:
            for param in m.parameters():
                param.requires_grad = True
        for param in model.panet.parameters():
            param.requires_grad = True
        for param in model.detection_head.parameters():
            param.requires_grad = True
        print('✓ Stage 1: Training backbone (layer3,4) + fusion + PANet + head')
        print('  Frozen: GFEM, backbone layer0/1/2')
    elif stage == 2:
        # Unfreeze only GFEM
        for param in model.dual_stream.rgb_gfem.parameters():
            param.requires_grad = True
        for param in model.dual_stream.thermal_gfem.parameters():
            param.requires_grad = True
        # Also train fusion_conv (small, connects GFEM to backbone)
        for param in model.dual_stream.rgb_fusion_conv.parameters():
            param.requires_grad = True
        for param in model.dual_stream.thermal_fusion_conv.parameters():
            param.requires_grad = True
        print('✓ Stage 2: Training only GFEM + fusion_conv')
        print('  Frozen: Backbone, fusion modules, PANet, detection head')
    else:  # stage == 3
        unfreeze_all_layers(model)
        print('✓ Stage 3: Full fine-tuning (all layers)')


def validate(model, dataloader, device, epoch, num_classes, conf_threshold=0.1, nms_threshold=0.5):
    """Validate model"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_detections = 0
    total_images = 0
    pred_label_hist = torch.zeros(num_classes, dtype=torch.long)
    gt_label_hist = torch.zeros(num_classes, dtype=torch.long)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    
    with torch.no_grad():
        for batch_data in progress_bar:
            if len(batch_data) == 3:
                # Paired RGB-Thermal
                rgb_images, thermal_images, targets = batch_data
                rgb_images = [img.to(device, non_blocking=True) for img in rgb_images]
                thermal_images = [img.to(device, non_blocking=True) for img in thermal_images]
                
                # Get predictions with post-processing
                predictions = model.predict(
                    rgb_images,
                    thermal_images,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
            else:
                # Single modality
                images, targets = batch_data
                images = [img.to(device, non_blocking=True) for img in images]
                thermal_images = images
                
                # Get predictions with post-processing
                predictions = model.predict(
                    images,
                    thermal_images,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
            
            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            total_detections += sum(int(p['boxes'].shape[0]) for p in predictions)
            total_images += len(predictions)
            for p in predictions:
                if p['labels'].numel() > 0:
                    pred_label_hist += torch.bincount(
                        p['labels'].detach().cpu(), minlength=num_classes
                    )[:num_classes]
            for t in targets:
                if t['labels'].numel() > 0:
                    gt_label_hist += torch.bincount(
                        t['labels'].detach().cpu(), minlength=num_classes
                    )[:num_classes]
    
    # Compute metrics
    from utils.metrics import calculate_map, calculate_ap50
    
    # Pass tensors on CPU to calculate_map (it uses torch.cat and torch ops)
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
    map_score = calculate_map(formatted_predictions, formatted_targets, num_classes)
    ap50_score = calculate_ap50(formatted_predictions, formatted_targets, num_classes)
    
    avg_detections = (total_detections / max(total_images, 1))
    return map_score, ap50_score, avg_detections, pred_label_hist, gt_label_hist


def main():
    parser = argparse.ArgumentParser(description='Train Fusion-YOLOv11')
    parser.add_argument('--dataset', type=str, default='dronergbt',
                       choices=['dronergbt', 'hituav', 'smod', 'combined', 'combined_all'],
                       help='Dataset: combined_all = HIT-UAV + DroneRGBT + SMOD in one cmd')
    parser.add_argument('--data_dir', type=str, 
                       default='sggf_net/data/dronergbt',
                       help='Dataset root directory (for single dataset)')
    parser.add_argument('--dronergbt_dir', type=str,
                       default='sggf_net/data/DroneRGBT',
                       help='DroneRGBT dataset directory (for combined training)')
    parser.add_argument('--smod_dir', type=str,
                       default='sggf_net/data/SMOD',
                       help='SMOD dataset directory (for combined training)')
    parser.add_argument('--hituav_dir', type=str,
                       default='data/hit-uav',
                       help='HIT-UAV dataset directory (data/hit-uav or data/HIT-UAV)')
    parser.add_argument('--dronergbt_subset_ratio', type=float, default=1.0,
                       help='Use only this fraction of DroneRGBT (0.0-1.0). E.g. 0.5 for 50%%.')
    parser.add_argument('--smod_subset_ratio', type=float, default=1.0,
                       help='Use only this fraction of SMOD (0.0-1.0). E.g. 0.25 for 25%%.')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (DroneRGBT: 2=background+person, SMOD: 3=background+person+vehicle, HIT-UAV: 6)')
    parser.add_argument('--checkpoint_dir', type=str, default='sggf_net/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8, increase if GPU memory allows)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4, increase for faster loading)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='Number of batches to prefetch per worker (default: 2)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5 for transfer learning, use 1e-4 for full training)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (path to .pth file, or "latest" or "best" to auto-detect)')
    parser.add_argument('--auto_resume', action='store_true',
                       help='Automatically resume from latest.pth if it exists')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value (default: 1.0, set to 0 to disable)')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze early ResNet layers (layer0, layer1, layer2) for faster training')
    parser.add_argument('--freeze_epochs', type=int, default=None,
                       help='Number of epochs to train with frozen backbone, then unfreeze (progressive unfreezing)')
    parser.add_argument('--stage', type=int, default=None, choices=[1, 2, 3],
                       help='Stage-by-stage training (faster): 1=backbone+fusion+head, 2=GFEM only, 3=full fine-tune. Use --resume to chain stages.')
    parser.add_argument('--max_size', type=int, default=640,
                       help='Max image size for resize (default 640 for Colab T4, use 1024/1536 if GPU has more memory)')
    parser.add_argument('--val_conf_threshold', type=float, default=0.1,
                       help='Validation confidence threshold (lower for early training diagnostics)')
    parser.add_argument('--val_nms_threshold', type=float, default=0.5,
                       help='Validation NMS IoU threshold')
    
    args = parser.parse_args()
    
    # combined_all defaults: 100% HIT-UAV, 50% DroneRGBT, 25% SMOD (override defaults if not explicitly set)
    argv_str = ' '.join(sys.argv)
    if args.dataset == 'combined_all':
        if '--dronergbt_subset_ratio' not in argv_str:
            args.dronergbt_subset_ratio = 0.5
        if '--smod_subset_ratio' not in argv_str:
            args.smod_subset_ratio = 0.25
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Optimize CUDA settings for faster training
    if device.type == 'cuda':
        # Enable cuDNN benchmark for consistent input sizes (faster convolutions)
        torch.backends.cudnn.benchmark = True
        # Enable deterministic mode only if reproducibility is needed (slower)
        # torch.backends.cudnn.deterministic = False
        print('✓ CUDA optimizations enabled (cuDNN benchmark)')
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset (max_size 640 = Colab T4 safe, reduces OOM and speeds up training)
    train_transform = get_train_transform(max_size=args.max_size)
    val_transform = get_val_transform(max_size=args.max_size)
    
    # Set num_classes based on dataset if not explicitly provided
    if args.num_classes == 2:  # Default value
        if args.dataset == 'smod':
            args.num_classes = 3  # SMOD: background + person + vehicle
        elif args.dataset == 'dronergbt':
            args.num_classes = 2  # DroneRGBT: background + person
        elif args.dataset == 'combined':
            args.num_classes = 3  # Combined: background + person + vehicle
        elif args.dataset == 'combined_all':
            args.num_classes = 3  # person + vehicle (HIT-UAV maps Car/Bicycle/OtherVehicle→vehicle, DontCare→skip)
        elif args.dataset == 'hituav':
            args.num_classes = 3  # HIT-UAV: person + vehicle (Person→person, Car/Bicycle/OtherVehicle→vehicle)
    
    if args.dataset == 'dronergbt':
        train_dataset = DroneRGBTDataset(
            root_dir=args.data_dir, split='train', transform=train_transform
        )
        val_dataset = DroneRGBTDataset(
            root_dir=args.data_dir, split='val', transform=val_transform
        )
        collate_fn = collate_fn_paired
    elif args.dataset == 'hituav':
        train_dataset = HITUAVDataset(
            root_dir=args.data_dir, split='train', transform=train_transform, use_person_vehicle=True
        )
        val_dataset = HITUAVDataset(
            root_dir=args.data_dir, split='val', transform=val_transform, use_person_vehicle=True
        )
        collate_fn = collate_fn_single
    elif args.dataset == 'smod':
        train_dataset = SMODDataset(
            root_dir=args.data_dir, split='train', transform=train_transform
        )
        val_dataset = SMODDataset(
            root_dir=args.data_dir, split='val', transform=val_transform
        )
        if args.smod_subset_ratio < 1.0:
            rng = torch.Generator().manual_seed(42)
            n_train = len(train_dataset)
            n_val = len(val_dataset)
            keep_train = max(1, int(n_train * args.smod_subset_ratio))
            keep_val = max(1, int(n_val * args.smod_subset_ratio))
            train_dataset = Subset(train_dataset, torch.randperm(n_train, generator=rng)[:keep_train].tolist())
            val_dataset = Subset(val_dataset, torch.randperm(n_val, generator=rng)[:keep_val].tolist())
            print(f"SMOD subset: {keep_train}/{n_train} train, {keep_val}/{n_val} val ({args.smod_subset_ratio*100:.0f}%)")
        collate_fn = collate_fn_single
    elif args.dataset == 'combined':
        # Combined dataset: DroneRGBT + SMOD
        print("Loading combined dataset (DroneRGBT + SMOD)...")
        
        # Check if directories exist
        if not os.path.exists(args.dronergbt_dir):
            raise ValueError(f"DroneRGBT directory not found: {args.dronergbt_dir}\n"
                           f"Please ensure DroneRGBT dataset is preprocessed and exists at this path.")
        
        if not os.path.exists(args.smod_dir):
            raise ValueError(f"SMOD directory not found: {args.smod_dir}\n"
                           f"Please ensure SMOD dataset exists at this path.\n"
                           f"Expected structure:\n"
                           f"  {args.smod_dir}/\n"
                           f"    ├── images/\n"
                           f"    │   ├── train/\n"
                           f"    │   ├── val/\n"
                           f"    │   └── test/\n"
                           f"    └── labels/\n"
                           f"        ├── train/\n"
                           f"        ├── val/\n"
                           f"        └── test/\n"
                           f"\nIf you only want to train on DroneRGBT, use:\n"
                           f"  --dataset dronergbt --data_dir {args.dronergbt_dir}")
        
        # Load DroneRGBT dataset
        try:
            dronergbt_train = DroneRGBTDataset(
                root_dir=args.dronergbt_dir, split='train', transform=train_transform
            )
            dronergbt_val = DroneRGBTDataset(
                root_dir=args.dronergbt_dir, split='val', transform=val_transform
            )
        except Exception as e:
            raise ValueError(f"Failed to load DroneRGBT dataset: {e}\n"
                           f"Please ensure DroneRGBT dataset is preprocessed correctly.")
        
        # Optionally use only a subset of DroneRGBT
        if args.dronergbt_subset_ratio < 1.0:
            rng = torch.Generator().manual_seed(42)
            n_train = len(dronergbt_train)
            n_val = len(dronergbt_val)
            keep_train = max(1, int(n_train * args.dronergbt_subset_ratio))
            keep_val = max(1, int(n_val * args.dronergbt_subset_ratio))
            idx_train = torch.randperm(n_train, generator=rng)[:keep_train].tolist()
            idx_val = torch.randperm(n_val, generator=rng)[:keep_val].tolist()
            dronergbt_train = Subset(dronergbt_train, idx_train)
            dronergbt_val = Subset(dronergbt_val, idx_val)
            print(f"  DroneRGBT subset: {keep_train}/{n_train} train, {keep_val}/{n_val} val ({args.dronergbt_subset_ratio*100:.0f}%)")
        
        # Load SMOD dataset
        try:
            smod_train = SMODDataset(
                root_dir=args.smod_dir, split='train', transform=train_transform
            )
            smod_val = SMODDataset(
                root_dir=args.smod_dir, split='val', transform=val_transform
            )
        except Exception as e:
            raise ValueError(f"Failed to load SMOD dataset: {e}\n"
                           f"Please ensure SMOD dataset structure is correct.\n"
                           f"Expected: {args.smod_dir}/images/train/ and {args.smod_dir}/labels/train/")
        
        # Optionally use only a subset of SMOD (for faster training)
        if args.smod_subset_ratio < 1.0:
            rng = torch.Generator().manual_seed(42)
            n_train = len(smod_train)
            n_val = len(smod_val)
            keep_train = max(1, int(n_train * args.smod_subset_ratio))
            keep_val = max(1, int(n_val * args.smod_subset_ratio))
            idx_train = torch.randperm(n_train, generator=rng)[:keep_train].tolist()
            idx_val = torch.randperm(n_val, generator=rng)[:keep_val].tolist()
            smod_train = Subset(smod_train, idx_train)
            smod_val = Subset(smod_val, idx_val)
            print(f"  SMOD subset: {keep_train}/{n_train} train, {keep_val}/{n_val} val ({args.smod_subset_ratio*100:.0f}%)")
        
        # Concatenate datasets
        train_dataset = ConcatDataset([dronergbt_train, smod_train])
        val_dataset = ConcatDataset([dronergbt_val, smod_val])
        
        collate_fn = collate_fn_combined
        
        print(f"✓ Combined dataset loaded:")
        print(f"  Train: {len(dronergbt_train)} DroneRGBT + {len(smod_train)} SMOD = {len(train_dataset)} total")
        print(f"  Val: {len(dronergbt_val)} DroneRGBT + {len(smod_val)} SMOD = {len(val_dataset)} total")
    elif args.dataset == 'combined_all':
        # Combined_all: full HIT-UAV + 50% DroneRGBT + 25% SMOD (or custom ratios via args)
        print("Loading combined_all dataset (HIT-UAV + DroneRGBT + SMOD)...")
        for name, path in [('HIT-UAV', args.hituav_dir), ('DroneRGBT', args.dronergbt_dir), ('SMOD', args.smod_dir)]:
            if not os.path.exists(path):
                raise ValueError(f"{name} directory not found: {path}\nSee PREPROCESSING.md for setup.")
        
        # Load HIT-UAV (full)
        try:
            hituav_train = HITUAVDataset(root_dir=args.hituav_dir, split='train', transform=train_transform, use_person_vehicle=True)
            hituav_val = HITUAVDataset(root_dir=args.hituav_dir, split='val', transform=val_transform, use_person_vehicle=True)
        except Exception as e:
            raise ValueError(f"Failed to load HIT-UAV: {e}\nRun: python scripts/preprocess_hituav.py --help")
        
        # Load DroneRGBT with subset
        dronergbt_train = DroneRGBTDataset(root_dir=args.dronergbt_dir, split='train', transform=train_transform)
        dronergbt_val = DroneRGBTDataset(root_dir=args.dronergbt_dir, split='val', transform=val_transform)
        if args.dronergbt_subset_ratio < 1.0:
            rng = torch.Generator().manual_seed(42)
            n_tr, n_v = len(dronergbt_train), len(dronergbt_val)
            dronergbt_train = Subset(dronergbt_train, torch.randperm(n_tr, generator=rng)[:max(1, int(n_tr * args.dronergbt_subset_ratio))].tolist())
            dronergbt_val = Subset(dronergbt_val, torch.randperm(n_v, generator=rng)[:max(1, int(n_v * args.dronergbt_subset_ratio))].tolist())
            print(f"  DroneRGBT subset: {len(dronergbt_train)}/{n_tr} train ({args.dronergbt_subset_ratio*100:.0f}%)")
        
        # Load SMOD with subset
        smod_train = SMODDataset(root_dir=args.smod_dir, split='train', transform=train_transform)
        smod_val = SMODDataset(root_dir=args.smod_dir, split='val', transform=val_transform)
        if args.smod_subset_ratio < 1.0:
            rng = torch.Generator().manual_seed(42)
            n_tr, n_v = len(smod_train), len(smod_val)
            smod_train = Subset(smod_train, torch.randperm(n_tr, generator=rng)[:max(1, int(n_tr * args.smod_subset_ratio))].tolist())
            smod_val = Subset(smod_val, torch.randperm(n_v, generator=rng)[:max(1, int(n_v * args.smod_subset_ratio))].tolist())
            print(f"  SMOD subset: {len(smod_train)}/{n_tr} train ({args.smod_subset_ratio*100:.0f}%)")
        
        train_dataset = ConcatDataset([hituav_train, dronergbt_train, smod_train])
        val_dataset = ConcatDataset([hituav_val, dronergbt_val, smod_val])
        collate_fn = collate_fn_combined
        print(f"✓ Combined_all loaded: {len(hituav_train)} HIT-UAV + {len(dronergbt_train)} DroneRGBT + {len(smod_train)} SMOD = {len(train_dataset)} train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Optimize DataLoader settings for faster data loading
    pin_memory = device.type == 'cuda'
    persistent_workers = args.num_workers > 0  # Keep workers alive between epochs
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True  # Drop last incomplete batch for consistent batch sizes
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    print(f'✓ DataLoader optimized: {args.num_workers} workers, prefetch={args.prefetch_factor}, pin_memory={pin_memory}')
    
    # Create model
    model = FusionYOLOv11(
        num_classes=args.num_classes,
        pretrained=True,
        fusion_type='concat_attention'
    )
    model = model.to(device)
    
    # Compile model for faster execution (PyTorch 2.0+)
    # DISABLED: torch.compile() is causing NaN losses with this model architecture
    # The CUDAGraph warnings and dynamic shapes are causing instability
    # Uncomment below if you want to try compilation (not recommended)
    # try:
    #     if hasattr(torch, 'compile'):
    #         print('✓ Compiling model with torch.compile() for faster execution...')
    #         model = torch.compile(model, mode='reduce-overhead')
    #         print('✓ Model compilation successful')
    # except Exception as e:
    #     print(f'⚠ Model compilation not available or failed: {e}')
    #     print('  Continuing without compilation...')
    print('⚠ Model compilation disabled (causes NaN losses with this architecture)')
    
    # Stage-by-stage OR transfer learning: freeze layers BEFORE creating optimizer
    if args.stage is not None:
        freeze_for_stage(model, args.stage)
        # Stage-specific epochs and learning rate (shorter = faster per stage)
        stage_epochs = {1: 30, 2: 20, 3: 30}  # Total ~80 epochs across 3 stages
        stage_lr = {1: 5e-5, 2: 1e-5, 3: 5e-6}
        args.epochs = stage_epochs.get(args.stage, args.epochs)
        args.lr = stage_lr.get(args.stage, args.lr)
        print(f'  Stage {args.stage}: {args.epochs} epochs, lr={args.lr}')
    elif args.freeze_backbone:
        freeze_backbone_layers(model, freeze_early=True)
    else:
        print('✓ Training all layers (no freezing)')
    
    # Optimizer: Only include trainable parameters (important for frozen layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler (fixed deprecation warning)
    if args.use_amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    # Resume from checkpoint
    start_epoch = 0
    best_map = 0.0
    
    # Auto-resume logic: if --auto_resume is set and latest.pth exists, use it
    if args.auto_resume and args.resume is None:
        latest_checkpoint = os.path.join(args.checkpoint_dir, 'latest.pth')
        if os.path.exists(latest_checkpoint):
            args.resume = latest_checkpoint
            print(f'Auto-resuming from: {latest_checkpoint}')
    
    # Handle resume shortcuts: "latest" or "best"
    if args.resume:
        if args.resume.lower() == 'latest':
            args.resume = os.path.join(args.checkpoint_dir, 'latest.pth')
        elif args.resume.lower() == 'best':
            args.resume = os.path.join(args.checkpoint_dir, 'best.pth')
    
    if args.resume:
        if not os.path.exists(args.resume):
            print(f'Warning: Checkpoint not found: {args.resume}')
            print('Starting training from scratch...')
        else:
            print(f'Loading checkpoint: {args.resume}')
            try:
                checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Note: When resuming with frozen layers or different stage, optimizer state may not match
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                print('  Warning: Optimizer state mismatch (possibly due to freezing/stage). Recreating optimizer...')
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=5e-4)
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception:
                    pass
            start_epoch = 0  # Start from 0 when chaining stages
            best_map = checkpoint.get('best_map', 0.0)
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print('  Loaded scaler state')
            print(f'  Resumed: Best mAP so far: {best_map:.4f}')
            print(f'  Continuing training from epoch 0 to {args.epochs}')
    
    # Training loop
    print(f'Starting training for {args.epochs} epochs...')
    if args.freeze_backbone and args.freeze_epochs:
        print(f'  Transfer Learning: Training with frozen backbone for {args.freeze_epochs} epochs')
        print(f'  Then unfreezing for remaining {args.epochs - args.freeze_epochs} epochs')
    
    for epoch in range(start_epoch, args.epochs):
        # Progressive unfreezing: unfreeze after freeze_epochs
        if args.freeze_backbone and args.freeze_epochs and epoch == args.freeze_epochs:
            print(f'\n🔄 Epoch {epoch+1}: Unfreezing all layers for fine-tuning...')
            unfreeze_all_layers(model)
            # Recreate optimizer with all parameters now that we've unfrozen
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
            # Adjust scheduler T_max for remaining epochs
            remaining_epochs = args.epochs - args.freeze_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            print(f'  Recreated optimizer with all parameters')
            print(f'  Adjusted scheduler for {remaining_epochs} remaining epochs')
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler, args.use_amp, args.grad_clip
        )
        
        # Validate
        val_map, val_ap50, avg_dets, pred_hist, gt_hist = validate(
            model,
            val_loader,
            device,
            epoch,
            args.num_classes,
            conf_threshold=args.val_conf_threshold,
            nms_threshold=args.val_nms_threshold
        )
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val mAP: {val_map:.4f}, Val AP50: {val_ap50:.4f}')
        print(f'  Val avg detections/image @conf={args.val_conf_threshold}: {avg_dets:.2f}')
        print(f'  Val pred label hist: {pred_hist.tolist()}')
        print(f'  Val gt label hist:   {gt_hist.tolist()}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
            'val_map': val_map,
            'val_ap50': val_ap50,
            'args': vars(args),
            'stage': args.stage
        }
        
        # Save scaler state if using AMP
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Save latest checkpoint (always)
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        print(f'  ✓ Saved latest checkpoint: {latest_path}')
        
        # Save best checkpoint (only when mAP improves)
        if val_map > best_map:
            best_map = val_map
            checkpoint['best_map'] = best_map
            best_path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'  ✓ Saved best model (mAP: {best_map:.4f}): {best_path}')
    
    print(f'Training completed! Best mAP: {best_map:.4f}')
    
    # Hint for stage chaining
    if args.stage is not None and args.stage < 3:
        next_stage = args.stage + 1
        best_path = os.path.join(args.checkpoint_dir, 'best.pth')
        print(f'\n💡 To continue with Stage {next_stage}, run:')
        extra = f'--dataset {args.dataset} --dronergbt_dir {args.dronergbt_dir} --smod_dir {args.smod_dir}'
        if args.dataset == 'combined_all':
            extra += f' --hituav_dir {args.hituav_dir}'
        print(f'   python scripts/train_fusion_yolov11.py --stage {next_stage} --resume {best_path} {extra}')


if __name__ == '__main__':
    main()
