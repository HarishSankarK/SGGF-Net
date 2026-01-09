"""
Evaluation script for SGGF-Net
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SGGFNet
from utils import VisDroneDataset, AITODDataset, HITUAVDataset, get_val_transform
from utils.metrics import calculate_map, calculate_ap50, calculate_precision_recall_f1


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def main():
    parser = argparse.ArgumentParser(description='Evaluate SGGF-Net')
    parser.add_argument('--dataset', type=str, default='visdrone', choices=['visdrone', 'aitod', 'hituav'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/hit-uav', help='Path to dataset root (default: data/hit-uav)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--max_size', type=int, default=1536, help='Max image size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Dataset
    val_transform = get_val_transform(max_size=args.max_size)
    
    if args.dataset == 'visdrone':
        dataset = VisDroneDataset(args.data_dir, split=args.split, transform=val_transform)
    elif args.dataset == 'aitod':
        dataset = AITODDataset(args.data_dir, split=args.split, transform=val_transform)
    else:  # hituav
        dataset = HITUAVDataset(args.data_dir, split=args.split, transform=val_transform, convert_to_rgb=True)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    
    # Model
    model = SGGFNet(num_classes=args.num_classes, pretrained=False)
    
    # Load checkpoint
    # weights_only=False is safe for our own checkpoints (contains numpy arrays)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    
    # Evaluate
    predictions = []
    targets_list = []
    
    print('Evaluating...')
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            predictions.extend(outputs)
            targets_list.extend(targets)
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx+1}/{len(dataloader)} batches')
    
    # Calculate metrics
    print('\nCalculating metrics...')
    print('  This may take a few minutes for large datasets...')
    mAP = calculate_map(predictions, targets_list, args.num_classes, verbose=True)
    print('\n  Calculating AP50...')
    AP50 = calculate_ap50(predictions, targets_list, args.num_classes, verbose=True)
    print('\n  Calculating Precision/Recall/F1...')
    precision, recall, f1 = calculate_precision_recall_f1(predictions, targets_list, args.num_classes)
    
    # Print results
    print('\n' + '='*50)
    print('Evaluation Results')
    print('='*50)
    print(f'mAP: {mAP:.4f}')
    print(f'AP50: {AP50:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('='*50)


if __name__ == '__main__':
    main()

