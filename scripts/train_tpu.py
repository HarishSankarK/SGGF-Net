"""
TPU Training script for SGGF-Net
Adapted for Google Cloud TPU v5e-1
"""

import os
import sys
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.optim.lr_scheduler as lr_scheduler

# TPU imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("⚠ Warning: torch_xla not available. Install with: pip install torch_xla[tpu]")

# Import model and utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.sggf_net import SGGFNet
from utils.dataset import VisDroneDataset, AITODDataset, HITUAVDataset
from utils.transforms import get_train_transform, get_val_transform
from utils.metrics import calculate_map, calculate_ap50, calculate_precision_recall_f1

# Try to import AMP (PyTorch 2.0+)
try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        USE_NEW_AMP = False
    except ImportError:
        autocast = None
        GradScaler = None
        USE_NEW_AMP = False


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, use_amp=False, 
                   max_grad_norm=10.0, grad_accum_steps=1):
    """Train for one epoch with TPU support"""
    model.train()
    total_loss = 0.0
    
    # TPU-specific: Mark step for TPU
    is_tpu = device.type == 'xla'
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device (TPU handles this automatically, but we do it for compatibility)
        if is_tpu:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        else:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        # Forward pass with mixed precision if enabled
        if use_amp and scaler is not None:
            # TPU uses different autocast
            if is_tpu:
                with autocast(device_type='cpu'):  # TPU uses CPU autocast
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            elif USE_NEW_AMP and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            elif not USE_NEW_AMP and device.type == 'cuda':
                with autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            # Standard precision
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # CRITICAL FIX: Skip batch if loss is NaN/Inf (prevents training crash)
        if not torch.isfinite(losses):
            print(f'⚠ Warning: NaN/Inf loss detected at batch {batch_idx+1}. Skipping this batch.')
            print(f'   Loss breakdown: {loss_dict}')
            optimizer.zero_grad(set_to_none=True)
            if is_tpu:
                xm.mark_step()  # TPU requires mark_step
            continue
        
        # Scale loss for gradient accumulation
        losses = losses / grad_accum_steps
        
        # Backward pass with gradient scaling
        if use_amp and scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()
        
        # Update optimizer only after accumulating gradients
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp and scaler is not None:
                # Gradient clipping before optimizer step
                if is_tpu:
                    # TPU: unscale and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # TPU: Mark step after optimizer update
            if is_tpu:
                xm.mark_step()
        
        total_loss += losses.item() * grad_accum_steps  # Scale back for reporting
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {losses.item() * grad_accum_steps:.4f}')
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train SGGF-Net on TPU')
    parser.add_argument('--dataset', type=str, default='visdrone', choices=['visdrone', 'aitod', 'hituav'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/hit-uav', help='Path to dataset root')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (TPU can handle larger batches)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--max_size', type=int, default=800, help='Max image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use Automatic Mixed Precision')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='Disable AMP')
    parser.add_argument('--val_freq', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--multi_scale', action='store_true', default=False, help='Enable multi-scale training')
    
    args = parser.parse_args()
    
    # TPU setup
    if TPU_AVAILABLE:
        device = xm.xla_device()
        print(f'✓ Using TPU device: {device}')
        print(f'✓ TPU cores: {xm.xla_world_size()}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'⚠ TPU not available, using device: {device}')
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f'Checkpoint directory: {args.checkpoint_dir}')
    
    # Dataset
    train_transform = get_train_transform(max_size=args.max_size, multi_scale=args.multi_scale)
    val_transform = get_val_transform(max_size=args.max_size)
    
    if args.dataset == 'visdrone':
        train_dataset = VisDroneDataset(args.data_dir, split='train', transform=train_transform)
        val_dataset = VisDroneDataset(args.data_dir, split='val', transform=val_transform)
    elif args.dataset == 'aitod':
        train_dataset = AITODDataset(args.data_dir, split='train', transform=train_transform)
        val_dataset = AITODDataset(args.data_dir, split='val', transform=val_transform)
    else:  # hituav
        train_dataset = HITUAVDataset(args.data_dir, split='train', transform=train_transform, convert_to_rgb=True)
        val_dataset = HITUAVDataset(args.data_dir, split='val', transform=val_transform, convert_to_rgb=True)
    
    # TPU-optimized DataLoader
    # TPUs benefit from more workers and specific settings
    num_workers = 8 if TPU_AVAILABLE else min(4, os.cpu_count() or 2)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=False,  # TPU doesn't use pin_memory
        prefetch_factor=2, persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # TPU benefits from fixed batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=False,
        prefetch_factor=2, persistent_workers=True if num_workers > 0 else False
    )
    
    # Wrap DataLoader for TPU
    if TPU_AVAILABLE:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)
        print('✓ Using TPU parallel data loader')
    
    # Model
    model = SGGFNet(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    # TPU: Move model to TPU device
    if TPU_AVAILABLE:
        model = xm.send_cpu_data_to_device(model, device)
        print('✓ Model moved to TPU')
    
    # Mixed precision training (AMP)
    scaler = None
    use_amp = args.use_amp and (device.type == 'xla' or device.type == 'cuda')
    if use_amp:
        if TPU_AVAILABLE:
            # TPU uses CPU autocast
            scaler = GradScaler() if GradScaler else None
        else:
            scaler = GradScaler() if GradScaler else None
        if scaler:
            print('✓ Using Automatic Mixed Precision (AMP) for faster training')
    
    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - Use CosineAnnealingLR with warmup
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f'✓ Using warmup ({args.warmup_epochs} epochs) + cosine annealing scheduler')
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f'Loading checkpoint from: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f'Resumed from epoch {start_epoch}')
        else:
            print(f'Checkpoint not found: {args.resume}')
    
    # Training loop
    best_map = 0.0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1, scaler, use_amp, 
                                    max_grad_norm=10.0, grad_accum_steps=args.grad_accum_steps)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Validate
        metrics = None
        if (epoch + 1) % args.val_freq == 0:
            print('Validating...')
            model.eval()
            predictions = []
            targets_list = []
            
            with torch.no_grad():
                for images, targets in val_loader:
                    if is_tpu:
                        images = [img.to(device) for img in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    else:
                        images = [img.to(device, non_blocking=True) for img in images]
                        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                    
                    outputs = model(images)
                    predictions.extend(outputs)
                    targets_list.extend(targets)
            
            # Calculate metrics
            mAP = calculate_map(predictions, targets_list, args.num_classes)
            AP50 = calculate_ap50(predictions, targets_list, args.num_classes)
            precision, recall, f1 = calculate_precision_recall_f1(predictions, targets_list, args.num_classes)
            
            metrics = {
                'mAP': mAP,
                'AP50': AP50,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f'Validation - mAP: {metrics["mAP"]:.4f}, AP50: {metrics["AP50"]:.4f}')
            model.train()
            
            # Save best model
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                if TPU_AVAILABLE:
                    xm.save(model.state_dict(), best_path)
                else:
                    torch.save(model.state_dict(), best_path)
                print(f'✓ Saved best model (mAP: {best_map:.4f})')
        else:
            print(f'Skipping validation (validate every {args.val_freq} epochs)')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        latest_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
        if TPU_AVAILABLE:
            xm.save(checkpoint, latest_path)
        else:
            torch.save(checkpoint, latest_path)
        
        # TPU: Mark step at end of epoch
        if TPU_AVAILABLE:
            xm.mark_step()
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()

