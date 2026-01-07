"""
TPU Training script for SGGF-Net (v5e-1 compatible)
- Robust TPU init (new + old APIs) with safe GPU/CPU fallback
- No global mutation of TPU_AVAILABLE (fixes UnboundLocalError)
- MpDeviceLoader and xm.save used only when on TPU
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.optim.lr_scheduler as lr_scheduler

# TPU imports (module-level flag only, never modified later)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
except Exception:
    TPU_AVAILABLE = False

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sggf_net import SGGFNet
from utils.dataset import VisDroneDataset, AITODDataset, HITUAVDataset
from utils.transforms import get_train_transform, get_val_transform
from utils.metrics import calculate_map, calculate_ap50, calculate_precision_recall_f1

# AMP imports (optional)
try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except Exception:
    try:
        from torch.cuda.amp import autocast, GradScaler
        USE_NEW_AMP = False
    except Exception:
        autocast, GradScaler, USE_NEW_AMP = None, None, False


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def train_one_epoch(model, dataloader, optimizer, device, epoch,
                    scaler=None, use_amp=False, max_grad_norm=10.0,
                    grad_accum_steps=1, is_tpu=False):
    """Train for one epoch with TPU support"""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        if is_tpu:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        else:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # Forward pass
        if use_amp and scaler is not None and device.type == 'cuda':
            if USE_NEW_AMP:
                with autocast(device_type='cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
            else:
                with autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Skip batch if loss is NaN/Inf
        if not torch.isfinite(losses):
            print(f'⚠ NaN/Inf at batch {batch_idx+1}, skipping. Loss dict: {loss_dict}')
            optimizer.zero_grad(set_to_none=True)
            if is_tpu:
                xm.mark_step()
            continue

        losses = losses / grad_accum_steps

        # Backward
        if use_amp and scaler is not None and device.type == 'cuda':
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # Step on accumulation boundary or last batch
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp and scaler is not None and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if is_tpu:
                xm.mark_step()

        total_loss += losses.item() * grad_accum_steps
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {losses.item() * grad_accum_steps:.4f}')

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train SGGF-Net on TPU')
    parser.add_argument('--dataset', type=str, default='hituav', choices=['visdrone', 'aitod', 'hituav'])
    parser.add_argument('--data_dir', type=str, default='data/hit-uav', help='Path to dataset root')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (TPU v5e-1: 16-32)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--max_size', type=int, default=800, help='Max image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use AMP')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='Disable AMP')
    parser.add_argument('--val_freq', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--multi_scale', action='store_true', default=False, help='Multi-scale training')

    args = parser.parse_args()

    # Local flags (do NOT modify module-level TPU_AVAILABLE)
    tpu_available = TPU_AVAILABLE
    is_tpu = False
    device = None

    # Try TPU init (new API then old API), otherwise fall back
    if tpu_available:
        try:
            # Try new API (torch_xla 2.9+)
            try:
                device = torch_xla.device()
                print(f'✓ Using TPU device (new API): {device}')
                is_tpu = True
                try:
                    world_size = xm.xla_world_size()
                    print(f'✓ TPU cores: {world_size}')
                except Exception:
                    try:
                        world_size = torch_xla._XLAC._xla_get_replication_devices_count()
                        print(f'✓ TPU cores: {world_size}')
                    except Exception:
                        print('✓ TPU initialized')
            except Exception as e_new:
                # Try old API
                try:
                    device = xm.xla_device()
                    print(f'✓ Using TPU device (old API): {device}')
                    is_tpu = True
                except Exception as e_old:
                    print('⚠ TPU init failed')
                    print(f'  New API: {str(e_new)[:120]}')
                    print(f'  Old API: {str(e_old)[:120]}')
                    tpu_available = False
        except Exception as e:
            print(f'⚠ TPU init exception: {str(e)[:200]}')
            tpu_available = False

    # Fallback to GPU/CPU
    if not tpu_available or device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'✓ Using device: {device}')
        is_tpu = False

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f'Checkpoint directory: {args.checkpoint_dir}')

    # Transforms
    train_transform = get_train_transform(max_size=args.max_size, multi_scale=args.multi_scale)
    val_transform = get_val_transform(max_size=args.max_size)

    # Datasets
    if args.dataset == 'visdrone':
        train_dataset = VisDroneDataset(args.data_dir, split='train', transform=train_transform)
        val_dataset = VisDroneDataset(args.data_dir, split='val', transform=val_transform)
    elif args.dataset == 'aitod':
        train_dataset = AITODDataset(args.data_dir, split='train', transform=train_transform)
        val_dataset = AITODDataset(args.data_dir, split='val', transform=val_transform)
    else:
        train_dataset = HITUAVDataset(args.data_dir, split='train', transform=train_transform, convert_to_rgb=True)
        val_dataset = HITUAVDataset(args.data_dir, split='val', transform=val_transform, convert_to_rgb=True)

    # DataLoaders
    num_workers = 8 if is_tpu else min(4, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=False,
        prefetch_factor=2, persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=False,
        prefetch_factor=2, persistent_workers=True if num_workers > 0 else False
    )

    # Wrap DataLoader for TPU
    if is_tpu and device.type == 'xla':
        try:
            train_loader = pl.MpDeviceLoader(train_loader, device)
            val_loader = pl.MpDeviceLoader(val_loader, device)
            print('✓ Using MpDeviceLoader')
        except Exception as e:
            print(f'⚠ MpDeviceLoader failed: {e}')
            print('⚠ Using standard DataLoader')

    # Model
    model = SGGFNet(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)

    # AMP (only for CUDA)
    scaler = GradScaler() if (args.use_amp and GradScaler is not None and device.type == 'cuda') else None
    if scaler:
        print('✓ Using AMP')

    # Optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # LR Scheduler with warmup
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f'✓ Warmup ({args.warmup_epochs}) + Cosine annealing')
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f'Loading checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    # Training loop
    best_map = 0.0
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch+1,
            scaler=scaler, use_amp=(scaler is not None),
            max_grad_norm=10.0, grad_accum_steps=args.grad_accum_steps,
            is_tpu=is_tpu
        )
        print(f'Train Loss: {train_loss:.4f}')

        scheduler.step()
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Validate
        metrics = None
        if (epoch + 1) % args.val_freq == 0:
            print('Validating...')
            model.eval()
            predictions, targets_list = [], []
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

            mAP = calculate_map(predictions, targets_list, args.num_classes)
            AP50 = calculate_ap50(predictions, targets_list, args.num_classes)
            precision, recall, f1 = calculate_precision_recall_f1(predictions, targets_list, args.num_classes)
            metrics = {'mAP': mAP, 'AP50': AP50, 'precision': precision, 'recall': recall, 'f1': f1}
            print(f'Val mAP: {mAP:.4f} AP50: {AP50:.4f} P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}')
            model.train()

            # Save best
            if mAP > best_map:
                best_map = mAP
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                try:
                    if is_tpu and device.type == 'xla':
                        xm.save(model.state_dict(), best_path)
                    else:
                        torch.save(model.state_dict(), best_path)
                    print(f'✓ Saved best model (mAP: {best_map:.4f})')
                except Exception as e:
                    print(f'⚠ Save best failed: {e}')
        else:
            print(f'Skipping validation (every {args.val_freq} epochs)')

        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pth')
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        if metrics is not None:
            ckpt['metrics'] = metrics

        try:
            if is_tpu and device.type == 'xla':
                xm.save(ckpt, latest_path)
            else:
                torch.save(ckpt, latest_path)
            print('✓ Saved latest checkpoint')
        except Exception as e:
            print(f'⚠ Save latest failed: {e}')

        if is_tpu:
            xm.mark_step()

    print('\nTraining completed!')


if __name__ == '__main__':
    main()
