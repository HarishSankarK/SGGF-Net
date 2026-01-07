"""
Training script for SGGF-Net
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.optim.lr_scheduler as lr_scheduler

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


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {losses.item():.4f}')
    
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
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--max_size', type=int, default=1536, help='Max image size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory (use Drive path in Colab)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (can be Drive path)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Create checkpoint directory (works for both local and Drive paths)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f'Checkpoint directory: {args.checkpoint_dir}')
    
    # Device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Dataset
    train_transform = get_train_transform(max_size=args.max_size)
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
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    
    # Model
    model = SGGFNet(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        metrics = validate(model, val_loader, device, args.num_classes)
        print(f'Validation - mAP: {metrics["mAP"]:.4f}, AP50: {metrics["AP50"]:.4f}')
        print(f'Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))
        
        # Save best
        if metrics['mAP'] > best_map:
            best_map = metrics['mAP']
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f'New best mAP: {best_map:.4f}')
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()

