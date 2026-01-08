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
        # Wrap in try-except for TPU to handle compilation errors gracefully
        try:
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
        except RuntimeError as e:
            error_str = str(e).lower()
            if is_tpu and ('compile' in error_str or 'sigterm' in error_str or 'to_cpu' in error_str or 'xla' in error_str):
                print(f'⚠ TPU compilation/runtime error at batch {batch_idx+1}: {str(e)[:200]}')
                print('   This may happen during first batch compilation - trying to recover...')
                if is_tpu:
                    try:
                        xm.mark_step()
                    except:
                        pass
                # Skip this batch and continue
                continue
            else:
                raise  # Re-raise if not a TPU compilation error

        # Skip batch if loss is NaN/Inf
        # For TPU, we need to check finiteness carefully
        if is_tpu:
            # On TPU, use xm.mesh_reduce or check differently
            # For now, just check if the value is finite after marking step
            try:
                is_finite = torch.isfinite(losses)
                # Get the actual boolean value (this should work)
                if not is_finite.item():
                    print(f'⚠ NaN/Inf at batch {batch_idx+1}, skipping.')
                    optimizer.zero_grad(set_to_none=True)
                    xm.mark_step()
                    continue
            except Exception:
                # If check fails, assume it's finite and continue
                pass
        else:
            if not torch.isfinite(losses):
                print(f'⚠ NaN/Inf at batch {batch_idx+1}, skipping. Loss dict: {loss_dict}')
                optimizer.zero_grad(set_to_none=True)
                continue

        losses = losses / grad_accum_steps

        # Backward
        if use_amp and scaler is not None and device.type == 'cuda':
            scaler.scale(losses).backward()
        else:
            losses.backward()

            # Step on accumulation boundary or last batch
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp and scaler is not None and not is_tpu and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # For TPU, standard optimizer.step() works with mark_step()
                if is_tpu:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if is_tpu:
                xm.mark_step()

        # Accumulate loss (need to sync for TPU before calling .item())
        # Note: mark_step() was already called after optimizer step if we did a step
        # But if we didn't step (gradient accumulation), we need to sync before .item()
        if is_tpu and (batch_idx + 1) % grad_accum_steps != 0:
            # Only mark step if we haven't stepped yet (accumulating gradients)
            xm.mark_step()
        
        # Get loss value - for TPU, ensure computation is synced before .item()
        if is_tpu:
            # For TPU, we need to wait for the computation to complete
            # mark_step() ensures computation is executed, then we can safely call .item()
            # Note: We must call .item() on the TPU tensor itself, not after moving to CPU
            try:
                # Ensure the loss tensor is computed by calling mark_step if needed
                # Then get the value - this should work after mark_step
                loss_value = float(losses.item()) * grad_accum_steps
            except Exception as e:
                # If .item() fails, try using xm.mesh_reduce to get the value
                print(f'⚠ Warning: Could not get loss value directly: {e}')
                loss_value = 0.0  # Use 0.0 as fallback, actual value will be tracked by optimizer
        else:
            loss_value = losses.item() * grad_accum_steps
        
        total_loss += loss_value
        
        if (batch_idx + 1) % 10 == 0:
            if is_tpu:
                # For TPU, ensure we print after mark_step
                try:
                    print_loss = float(losses.item()) * grad_accum_steps
                    print(f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {print_loss:.4f}')
                except Exception:
                    # If we can't get the value, just print that we're training
                    print(f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] Training...')
            else:
                print(f'Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss_value:.4f}')

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train SGGF-Net on TPU')
    parser.add_argument('--dataset', type=str, default='hituav', choices=['visdrone', 'aitod', 'hituav'])
    parser.add_argument('--data_dir', type=str, default='data/hit-uav', help='Path to dataset root')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes (HIT-UAV: 6, VisDrone: 11, AI-TOD: 9)')
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
    # Note: If TPU is already initialized, we'll reuse it instead of reinitializing
    if tpu_available:
        try:
            # Check if TPU environment is available first
            import os
            tpu_addr = os.environ.get('COLAB_TPU_ADDR', '')
            
            # Try new API (torch_xla 2.9+) - this gets existing device if already initialized
            # Note: Even if COLAB_TPU_ADDR is not set, TPU might still be available
            if not tpu_addr:
                print('⚠ COLAB_TPU_ADDR not set, but checking if TPU is available...')
            
            try:
                # Use device() which gets the device without forcing reinit
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
                    # Try old API as fallback
                    try:
                        device = xm.xla_device()
                        print(f'✓ Using TPU device (old API): {device}')
                        is_tpu = True
                    except Exception as e_old:
                        # If device is busy, try multiple strategies to recover
                        error_str = str(e_old)
                        if 'Device or resource busy' in error_str or 'iommu' in error_str.lower():
                            print('⚠ TPU device busy, attempting recovery...')
                            import time
                            
                            # Strategy 1: Wait and retry with new API
                            print('  Strategy 1: Waiting 3 seconds and retrying with new API...')
                            time.sleep(3)
                            try:
                                device = torch_xla.device()
                                print(f'✓ TPU device acquired (new API after wait): {device}')
                                is_tpu = True
                            except Exception:
                                # Strategy 2: Wait longer and try old API
                                print('  Strategy 2: Waiting 5 more seconds and trying old API...')
                                time.sleep(5)
                                try:
                                    device = xm.xla_device()
                                    print(f'✓ TPU device acquired (old API after wait): {device}')
                                    is_tpu = True
                                except Exception as e_retry:
                                    print('⚠ TPU init failed after all retry strategies')
                                    print(f'  Error: {str(e_retry)[:120]}')
                                    print('  💡 Suggestion: Restart Colab runtime (Runtime → Restart runtime)')
                                    print('  💡 Then re-run Cell 2 and Cell 5')
                                    tpu_available = False
                        else:
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
    # Check if device is XLA device (hasattr check works for both old and new APIs)
    if is_tpu:
        try:
            # Check if device is XLA (either by type or by checking if it's an XLA device)
            is_xla_device = hasattr(device, 'index') or str(device).startswith('xla:') or 'xla' in str(type(device)).lower()
            if is_xla_device:
                train_loader = pl.MpDeviceLoader(train_loader, device)
                val_loader = pl.MpDeviceLoader(val_loader, device)
                print('✓ Using MpDeviceLoader')
            else:
                print(f'⚠ Device {device} is not recognized as XLA device')
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
        # Load checkpoint - use xm.load for TPU, torch.load for CPU/GPU
        if is_tpu:
            try:
                ckpt = xm.load(args.resume)
            except Exception:
                # Fallback to torch.load if xm.load fails (e.g., checkpoint saved on GPU)
                ckpt = torch.load(args.resume, map_location='cpu')
        else:
            ckpt = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    # Warmup forward pass for TPU (compiles the graph before training)
    # This is critical to avoid SIGTERM crashes during first batch
    if is_tpu:
        print('\n🔥 Warming up TPU (compiling graph)...')
        print('   This may take 1-2 minutes - TPU is compiling the computation graph')
        print('   Please be patient - compilation is a one-time cost per session')
        model.train()
        warmup_success = False
        try:
            # Get a dummy batch to compile the graph
            dummy_batch = next(iter(train_loader))
            images, targets = dummy_batch
            
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass to trigger compilation (use no_grad to avoid gradient computation)
            with torch.no_grad():
                loss_dict = model(images, targets)
                # Just access the dict to trigger computation, don't sum (might trigger CPU access)
                _ = list(loss_dict.values())
            
            # Mark step to ensure compilation completes
            xm.mark_step()
            print('✓ TPU graph compiled successfully!')
            warmup_success = True
        except (RuntimeError, Exception) as e:
            error_str = str(e).lower()
            if 'sigterm' in error_str or 'to_cpu' in error_str or 'compile' in error_str:
                print(f'⚠ Warning: TPU warmup encountered compilation issue: {str(e)[:200]}')
                print('   This might be due to memory constraints or model complexity')
                print('   Training will attempt to continue, but may crash on first batch')
                print('   💡 Try reducing batch_size or max_size if this persists')
            else:
                print(f'⚠ Warning: TPU warmup failed: {str(e)[:200]}')
            warmup_success = False
            if is_tpu:
                try:
                    xm.mark_step()
                except:
                    pass
        
        if not warmup_success:
            print('   ⚠ Proceeding without warmup - first batch may trigger compilation')

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
                    
                    # Mark step for TPU during validation
                    if is_tpu:
                        xm.mark_step()

            # Sync TPU before metric calculation
            if is_tpu:
                xm.mark_step()
            
            mAP = calculate_map(predictions, targets_list, args.num_classes)
            AP50 = calculate_ap50(predictions, targets_list, args.num_classes)
            precision, recall, f1 = calculate_precision_recall_f1(predictions, targets_list, args.num_classes)
            metrics = {'mAP': mAP, 'AP50': AP50, 'precision': precision, 'recall': recall, 'f1': f1}
            print(f'Val mAP: {mAP:.4f} AP50: {AP50:.4f} P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}')
            model.train()

            # Save best (sync TPU before saving)
            if mAP > best_map:
                best_map = mAP
                best_path = os.path.join(args.checkpoint_dir, 'best.pth')
                try:
                    if is_tpu:
                        # Sync TPU before saving
                        xm.mark_step()
                        # Get state dict from all cores (only need rank 0, but sync first)
                        state_dict = model.state_dict()
                        # Save only on rank 0 to avoid conflicts
                        try:
                            # Try is_master_ordinal (new API)
                            is_master = xm.is_master_ordinal()
                        except AttributeError:
                            # Fallback to get_ordinal (old API)
                            is_master = (xm.get_ordinal() == 0)
                        if is_master:
                            xm.save(state_dict, best_path)
                        # Sync all cores after save
                        xm.mark_step()
                        if is_master:
                            print(f'✓ Saved best model (mAP: {best_map:.4f})')
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
            if is_tpu:
                # Sync TPU before saving
                xm.mark_step()
                # Save only on rank 0 to avoid conflicts
                try:
                    # Try is_master_ordinal (new API)
                    is_master = xm.is_master_ordinal()
                except AttributeError:
                    # Fallback to get_ordinal (old API)
                    is_master = (xm.get_ordinal() == 0)
                if is_master:
                    xm.save(ckpt, latest_path)
                # Sync all cores after save
                xm.mark_step()
                if is_master:
                    print('✓ Saved latest checkpoint')
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
