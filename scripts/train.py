"""
Training script for SGGF-Net
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
try:
    # PyTorch 2.0+ uses torch.amp
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SGGFNet
from utils import VisDroneDataset, AITODDataset, HITUAVDataset, get_train_transform, get_val_transform
from utils.metrics import calculate_map, calculate_ap50, calculate_precision_recall_f1


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, use_amp=False, 
                   max_grad_norm=10.0, grad_accum_steps=1):
    """Train for one epoch with optional mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device (non-blocking for faster transfer)
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        # Forward pass with mixed precision if enabled
        if use_amp and scaler is not None:
            # Use device-specific autocast (PyTorch 2.0+) or legacy autocast
            if USE_NEW_AMP and device.type == 'cuda':
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += losses.item() * grad_accum_steps  # Scale back for reporting
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {losses.item() * grad_accum_steps:.4f}')
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, num_classes):
    """Validate the model"""
    model.eval()
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            predictions.extend(outputs)
            targets_list.extend(targets)
    
    # Calculate metrics
    mAP = calculate_map(predictions, targets_list, num_classes)
    AP50 = calculate_ap50(predictions, targets_list, num_classes)
    precision, recall, f1 = calculate_precision_recall_f1(predictions, targets_list, num_classes)
    
    return {
        'mAP': mAP,
        'AP50': AP50,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Train SGGF-Net')
    parser.add_argument('--dataset', type=str, default='visdrone', choices=['visdrone', 'aitod', 'hituav'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/hit-uav', help='Path to dataset root (default: data/hit-uav)')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (default: 0.0005, optimized for stability)')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps (default: 1, use 2 for effective batch_size=4)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for learning rate (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--max_size', type=int, default=1024, 
                        help='Max image size (default: 1024, use 1536 only if you have enough GPU memory)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory (use Drive path in Colab)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (can be Drive path)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP) for faster training (default: True)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable AMP (use full precision)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile for faster training (PyTorch 2.0+, default: False)')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='Validate every N epochs (default: 5, set to 1 for every epoch)')
    parser.add_argument('--multi_scale', action='store_true', default=False,
                        help='Enable multi-scale training (randomly sample from [1024, 1280, 1536])')
    
    args = parser.parse_args()
    
    # Create checkpoint directory (works for both local and Drive paths)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f'Checkpoint directory: {args.checkpoint_dir}')
    
    # Device with CUDA verification
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print('⚠ CUDA requested but not available. Falling back to CPU.')
            print('⚠ To use GPU, run Step 2 in the notebook to install CUDA PyTorch.')
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            print(f'✓ Using device: {device}')
            print(f'✓ CUDA device: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device(args.device)
        print(f'Using device: {device}')
    
    # Dataset with optional multi-scale training
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
    
    # Optimized DataLoader settings for speed
    # Increase num_workers for faster data loading (Colab typically supports 2-4)
    num_workers = min(4, os.cpu_count() or 2)  # Use up to 4 workers
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        prefetch_factor=2, persistent_workers=True if num_workers > 0 else False
    )
    
    # Model
    model = SGGFNet(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    # Optional: Compile model for faster training (PyTorch 2.0+)
    # Note: torch.compile can cause OOM during compilation, so use with caution
    if args.compile and hasattr(torch, 'compile'):
        print('Compiling model with torch.compile for faster training...')
        print('⚠ Warning: torch.compile may cause OOM. If you get OOM errors, disable with --no_compile')
        try:
            # Use 'default' mode instead of 'reduce-overhead' to reduce memory usage during compilation
            model = torch.compile(model, mode='default')
            print('✓ Model compiled')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'OOM' in str(e):
                print('⚠ torch.compile failed due to OOM. Continuing without compilation.')
                print('   Training will still be fast with AMP enabled.')
            else:
                raise
    
    # Mixed precision training (AMP) - 1.5-2x speedup with minimal accuracy impact
    scaler = None
    use_amp = args.use_amp and device.type == 'cuda'
    if use_amp:
        # Use device-specific GradScaler for PyTorch 2.0+, legacy for older versions
        if USE_NEW_AMP and device.type == 'cuda':
            scaler = GradScaler(device='cuda')
        else:
            scaler = GradScaler()
        print('✓ Using Automatic Mixed Precision (AMP) for faster training')
    
    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - Use CosineAnnealingLR with warmup for better convergence
    if args.warmup_epochs > 0:
        # Warmup + Cosine annealing
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / args.warmup_epochs
            else:
                # Cosine annealing
                return 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f'✓ Using warmup ({args.warmup_epochs} epochs) + cosine annealing scheduler')
    else:
        # Fallback to step LR
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Resume from checkpoint (supports both local and Drive paths)
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f'Loading checkpoint from: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f'✓ Resumed from epoch {start_epoch}')
        else:
            print(f'⚠ Warning: Checkpoint not found at {args.resume}, starting from epoch 0')
    
    # Training loop
    best_map = 0.0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1, scaler, use_amp, 
                                    max_grad_norm=10.0, grad_accum_steps=args.grad_accum_steps)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate (less frequently to save time)
        metrics = None
        if (epoch + 1) % args.val_freq == 0 or (epoch + 1) == args.num_epochs:
            metrics = validate(model, val_loader, device, args.num_classes)
            print(f'Validation - mAP: {metrics["mAP"]:.4f}, AP50: {metrics["AP50"]:.4f}')
            print(f'Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}')
        else:
            print(f'Skipping validation (validate every {args.val_freq} epochs)')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss
        }
        
        # Add metrics if available
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))
        
        # Save best (only if metrics are available)
        if metrics is not None and metrics['mAP'] > best_map:
            best_map = metrics['mAP']
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f'New best mAP: {best_map:.4f}')
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()

