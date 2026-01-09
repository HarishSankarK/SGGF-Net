"""
Streamlined Training Script for SGGF-Net (M1 Optimized)
3-Stage Training: Baseline → GFEM → NDPA+ARPM
Target: ~60 minutes total training time
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SGGFNet
from utils import HITUAVDataset, get_train_transform, get_val_transform
from utils.metrics import calculate_map, calculate_ap50, calculate_precision_recall_f1


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def freeze_backbone_early_layers(model):
    """Freeze ResNet early layers (layer0, layer1, layer2) for transfer learning"""
    for name, param in model.backbone.named_parameters():
        if 'layer0' in name or 'layer1' in name or 'layer2' in name:
            param.requires_grad = False
    print("✓ Frozen ResNet layers: layer0, layer1, layer2")


def freeze_except(model, module_names):
    """Freeze all parameters except specified modules"""
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specified modules
    for name in module_names:
        for param in getattr(model, name).parameters():
            param.requires_grad = True
    print(f"✓ Training only: {', '.join(module_names)}")


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # MPS: non_blocking doesn't work well, disable it
    non_blocking = device.type == 'cuda'
    
    print(f'  Total batches: {len(dataloader)}')
    if device.type == 'cpu':
        print('  ⚠ CPU training is slow - please be patient!')
        print('  💡 For faster training, use Google Colab (T4 GPU)')
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Progress update (CPU is very slow, provide frequent updates)
        if device.type == 'cpu':
            if batch_idx == 0:
                print(f'  Loading first batch (batch 1/{len(dataloader)})...')
                print('  ⏳ This may take 1-2 minutes on CPU (please wait)')
            elif batch_idx < 3:
                print(f'  Processing batch {batch_idx+1}/{len(dataloader)}...')
        
        images = [img.to(device, non_blocking=non_blocking) for img in images]
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]
        
        if device.type == 'cpu' and batch_idx == 0:
            print('  ⏳ Running forward pass (this is the slowest part on CPU)...')
        
        optimizer.zero_grad()
        
        # Forward pass (MPS: no AMP due to memory issues)
        if use_amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
        else:
            # Standard precision (MPS or CPU)
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        
        if device.type == 'cpu' and batch_idx == 0:
            print(f'  ✓ Forward pass completed! Loss: {loss.item():.4f}')
            print('  ⚠ CPU training is ~10x slower than GPU. Consider using Colab for faster training.')
        
        # Skip NaN/Inf
        if not torch.isfinite(loss):
            print(f'⚠ Warning: NaN/Inf loss at batch {batch_idx+1}, skipping')
            continue
        
        # Backward
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
        # MPS: Sync and empty cache periodically
        if device.type == 'mps':
            if (batch_idx + 1) % 50 == 0:
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass  # Ignore cache errors on MPS
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress updates: more frequent on CPU (it's slow), less frequent on GPU
        if device.type == 'cpu':
            # CPU: Print every 10 batches or every batch for first 10
            if batch_idx < 10 or (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        else:
            # GPU: Print every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f'  Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device, num_classes):
    """Validate the model"""
    model.eval()
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device, non_blocking=True) for img in images]
            # Move targets to device as well
            targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            outputs = model(images)
            predictions.extend(outputs)
            targets_list.extend(targets)
    
    # Move all predictions to CPU for metric calculation (more memory efficient)
    # This ensures all tensors are on the same device
    predictions_cpu = []
    for pred in predictions:
        pred_cpu = {
            'boxes': pred['boxes'].cpu() if isinstance(pred['boxes'], torch.Tensor) else pred['boxes'],
            'labels': pred['labels'].cpu() if isinstance(pred['labels'], torch.Tensor) else pred['labels'],
            'scores': pred['scores'].cpu() if isinstance(pred['scores'], torch.Tensor) else pred['scores']
        }
        predictions_cpu.append(pred_cpu)
    
    targets_cpu = []
    for target in targets_list:
        target_cpu = {
            'boxes': target['boxes'].cpu() if isinstance(target['boxes'], torch.Tensor) else target['boxes'],
            'labels': target['labels'].cpu() if isinstance(target['labels'], torch.Tensor) else target['labels']
        }
        targets_cpu.append(target_cpu)
    
    # Calculate metrics (all tensors now on CPU)
    mAP = calculate_map(predictions_cpu, targets_cpu, num_classes, verbose=False)
    AP50 = calculate_ap50(predictions_cpu, targets_cpu, num_classes, verbose=False)
    precision, recall, f1 = calculate_precision_recall_f1(predictions_cpu, targets_cpu, num_classes)
    
    return {
        'mAP': mAP,
        'AP50': AP50,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def create_subset_dataset(dataset, subset_ratio=0.35, seed=42):
    """Create a random subset of the dataset"""
    total_size = len(dataset)
    subset_size = int(total_size * subset_ratio)
    
    random.seed(seed)
    np.random.seed(seed)
    indices = random.sample(range(total_size), subset_size)
    
    print(f"✓ Using {subset_size}/{total_size} samples ({subset_ratio*100:.0f}% of dataset)")
    return Subset(dataset, indices)


def main():
    parser = argparse.ArgumentParser(description='Train SGGF-Net (Staged Training for M1)')
    parser.add_argument('--data_dir', type=str, default='data/hit-uav', help='Dataset directory')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3], 
                       help='Training stage: 1=Baseline, 2=GFEM, 3=NDPA+ARPM')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--subset_ratio', type=float, default=0.35, help='Training subset ratio (0.35 = 35%%)')
    
    args = parser.parse_args()
    
    # Device setup - MPS has critical memory issues on M1, use CPU instead
    # MPS crashes with "Failed to allocate IOGPUDeviceShmem" - it's too unreliable
    # CPU is slower but stable (2-3 hours vs crashes)
    force_mps = os.environ.get('USE_MPS', '0').lower() in ['1', 'true', 'yes']
    
    if torch.backends.mps.is_available() and force_mps:
        # Only use MPS if explicitly requested (NOT RECOMMENDED - will likely crash)
        device = torch.device('mps')
        use_amp = False
        print('⚠⚠⚠ Using MPS (NOT RECOMMENDED - likely to crash!) ⚠⚠⚠')
        print('  MPS crashes with "IOGPUDeviceShmem" errors on complex models')
        print('  If you encounter crashes, remove USE_MPS=1 to use CPU')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        use_amp = True
        print('✓ Using CUDA GPU')
    else:
        device = torch.device('cpu')
        use_amp = False
        if torch.backends.mps.is_available():
            print('✓ Using CPU (MPS available but disabled due to memory crashes)')
            print('  MPS fails with "IOGPUDeviceShmem" errors on M1 - CPU is stable')
            print('  Training will take ~2-3 hours on CPU (but it will complete!)')
        else:
            print('✓ Using CPU')
    
    # Training configuration optimized for M1
    config = {
        'batch_size': 1,
        'max_size': 640,  # Reduced from 1536 for speed
        'num_epochs': 10,
        'lr': 1e-4,
        'rpn_post_nms_top_n': 300,  # Reduced from 1000 for speed
    }
    
    print("\n" + "="*70, flush=True)
    print(f"STAGE {args.stage} TRAINING", flush=True)
    print("="*70, flush=True)
    
    if args.stage == 1:
        print("Stage 1: Baseline Faster-RCNN (Backbone + FPN + RPN)")
        print("  - NO GFEM, NO NDPA, NO ARPM")
        print("  - Purpose: Stable anchor learning")
        config['num_epochs'] = 8  # Shorter for baseline
    elif args.stage == 2:
        print("Stage 2: Enable GFEM only")
        print("  - Freeze everything except GFEM")
        config['num_epochs'] = 6
        config['lr'] = 5e-5  # Lower LR for fine-tuning
    else:
        print("Stage 3: Enable NDPA + ARPM")
        print("  - Short fine-tuning of attention modules")
        config['num_epochs'] = 4
        config['lr'] = 1e-5  # Very low LR
    
    print("\nConfiguration:", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Batch size: {config['batch_size']}", flush=True)
    print(f"  Image size: {config['max_size']}", flush=True)
    print(f"  Epochs: {config['num_epochs']}", flush=True)
    print(f"  Learning rate: {config['lr']}", flush=True)
    print(f"  Dataset subset: {args.subset_ratio*100:.0f}%", flush=True)
    print(f"  Mixed precision: {use_amp}", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Dataset - Check if it exists first
    if not os.path.exists(args.data_dir):
        print(f"\n❌ ERROR: Dataset directory not found: {args.data_dir}")
        print("\nPlease ensure the HIT-UAV dataset is available:")
        print("  1. Download from: https://github.com/Syo9/HIT-UAV")
        print("  2. Extract to: data/hit-uav/")
        print("  3. Expected structure:")
        print("     data/hit-uav/")
        print("       ├── images/")
        print("       └── annotations/")
        sys.exit(1)
    
    # Dataset
    try:
        train_transform = get_train_transform(max_size=config['max_size'])
        val_transform = get_val_transform(max_size=config['max_size'])
        
        print(f"\nLoading dataset from: {args.data_dir}", flush=True)
        train_dataset = HITUAVDataset(args.data_dir, split='train', transform=train_transform, convert_to_rgb=True)
        val_dataset = HITUAVDataset(args.data_dir, split='val', transform=val_transform, convert_to_rgb=True)
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples", flush=True)
    except Exception as e:
        print(f"\n❌ ERROR loading dataset: {e}")
        print(f"\nPlease check:")
        print(f"  1. Dataset directory exists: {args.data_dir}")
        print(f"  2. Dataset structure is correct (images/ and annotations/ folders)")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create subset for faster training
    train_subset = create_subset_dataset(train_dataset, subset_ratio=args.subset_ratio)
    
    # MPS: Disable pin_memory and reduce workers (MPS has memory issues)
    num_workers = 0 if device.type == 'mps' else 2
    pin_memory = False if device.type == 'mps' else (device.type == 'cuda')
    
    train_loader = DataLoader(
        train_subset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory
    )
    
    # Model - Create with reduced RPN proposals
    print("Creating model...", flush=True)
    model = SGGFNet(
        num_classes=args.num_classes, 
        pretrained=True,
        rpn_post_nms_top_n=config['rpn_post_nms_top_n']
    )
    print("✓ Model created", flush=True)
    
    # Stage-specific model configuration
    if args.stage == 1:
        # Baseline: Freeze GFEM, NDPA, ARPM (only train Backbone+FPN+RPN+ROIHead)
        for param in model.gfem.parameters():
            param.requires_grad = False
        for param in model.ndpa.parameters():
            param.requires_grad = False
        for param in model.arpm.parameters():
            param.requires_grad = False
        # Freeze early backbone layers for transfer learning
        freeze_backbone_early_layers(model)
        # Train: Backbone (layer3, layer4), FPN, RPN, ROI Head, fusion_conv
        print("✓ Stage 1: Training Backbone (layer3, layer4) + FPN + RPN + ROI Head", flush=True)
        print("  (GFEM, NDPA, ARPM are frozen)", flush=True)
        
    elif args.stage == 2:
        # Stage 2: Only train GFEM, keep everything else frozen
        freeze_except(model, ['gfem', 'fusion_conv'])  # fusion_conv needs updating too
        print("✓ Stage 2: Training only GFEM module + fusion layer", flush=True)
        
    else:  # Stage 3
        # Stage 3: Train NDPA and ARPM, keep everything else frozen
        freeze_except(model, ['ndpa', 'arpm'])
        print("✓ Stage 3: Training only NDPA and ARPM modules", flush=True)
    
    print(f"Moving model to device: {device}...", flush=True)
    model = model.to(device)
    print("✓ Model moved to device", flush=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}", flush=True)
        # weights_only=False is safe for our own checkpoints
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_map = checkpoint.get('metrics', {}).get('mAP', 0.0)
        print(f"  Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}\n", flush=True)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=1e-4
    )
    
    # Mixed precision scaler
    if use_amp:
        if device.type == 'cuda':
            scaler = torch.amp.GradScaler('cuda')
        elif device.type == 'mps':
            # MPS: Disable AMP for now (has memory issues)
            scaler = None
            use_amp = False
            print("⚠ MPS: Mixed precision disabled (memory issues)")
        else:
            scaler = None
            use_amp = False
    else:
        scaler = None
    
    # MPS: Add memory management (if using MPS)
    if device.type == 'mps':
        print("⚠ MPS: Using single worker and no pin_memory for stability")
        print("⚠ WARNING: MPS may still crash on large models. Consider using CPU for stability.")
        # Sync before starting (helps with MPS initialization)
        try:
            torch.mps.synchronize()
        except Exception as e:
            print(f"⚠ MPS initialization failed: {e}")
            print("  Falling back to CPU...")
            device = torch.device('cpu')
            use_amp = False
    
    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\nStarting training for {config['num_epochs']} epochs...\n", flush=True)
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]", flush=True)
        print("-" * 70, flush=True)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler, use_amp)
        print(f"Train Loss: {train_loss:.4f}\n")
        
        # Validate every 2 epochs or at the end
        metrics = None
        is_new_best = False
        if (epoch + 1) % 2 == 0 or (epoch + 1) == config['num_epochs']:
            print("Validating...", flush=True)
            metrics = validate(model, val_loader, device, args.num_classes)
            
            print(f"Validation Metrics:", flush=True)
            print(f"  mAP: {metrics['mAP']:.4f}", flush=True)
            print(f"  AP50: {metrics['AP50']:.4f}", flush=True)
            print(f"  Precision: {metrics['precision']:.4f}", flush=True)
            print(f"  Recall: {metrics['recall']:.4f}", flush=True)
            print(f"  F1: {metrics['f1']:.4f}\n", flush=True)
            
            # Update best mAP if validation ran
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                is_new_best = True
        
        # Save checkpoint (always save latest, best only if metrics exist)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'metrics': metrics if metrics is not None else {},
            'stage': args.stage
        }
        
        # Save latest (always)
        latest_path = os.path.join(args.checkpoint_dir, f'stage{args.stage}_latest.pth')
        torch.save(checkpoint, latest_path)
        if os.path.exists(latest_path):
            print(f"✓ Saved latest checkpoint (epoch {epoch+1}): {latest_path}\n", flush=True)
        else:
            print(f"⚠ WARNING: Checkpoint file not found after save: {latest_path}\n", flush=True)
        
        # Save best only if validation ran and we have a new best mAP
        if is_new_best:
            best_path = os.path.join(args.checkpoint_dir, f'stage{args.stage}_best.pth')
            torch.save(checkpoint, best_path)
            if os.path.exists(best_path):
                print(f"✓ Saved best checkpoint (mAP: {best_map:.4f}): {best_path}\n", flush=True)
            else:
                print(f"⚠ WARNING: Best checkpoint file not found after save: {best_path}\n", flush=True)
    
    # Ensure best checkpoint exists (use latest if no best was saved)
    best_checkpoint_path = os.path.join(args.checkpoint_dir, f'stage{args.stage}_best.pth')
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, f'stage{args.stage}_latest.pth')
    
    if not os.path.exists(best_checkpoint_path):
        if os.path.exists(latest_checkpoint_path):
            # If no best checkpoint was saved (validation didn't improve), copy latest as best
            import shutil
            shutil.copy2(latest_checkpoint_path, best_checkpoint_path)
            if os.path.exists(best_checkpoint_path):
                print(f"✓ Created best checkpoint from latest (mAP: {best_map:.4f})", flush=True)
            else:
                print(f"⚠ WARNING: Failed to create best checkpoint from latest", flush=True)
        else:
            print(f"⚠ WARNING: No checkpoints found! Latest: {latest_checkpoint_path}, Best: {best_checkpoint_path}", flush=True)
    else:
        print(f"✓ Best checkpoint exists: {best_checkpoint_path}", flush=True)
    
    print("="*70)
    print(f"Stage {args.stage} training completed!")
    print(f"Best mAP: {best_map:.4f}")
    print("="*70)
    
    if args.stage < 3:
        next_stage = args.stage + 1
        best_checkpoint = os.path.join(args.checkpoint_dir, f'stage{args.stage}_best.pth')
        print(f"\n💡 To continue with Stage {next_stage}, run:")
        print(f"   python scripts/train.py --stage {next_stage} --resume {best_checkpoint}")


if __name__ == '__main__':
    main()
