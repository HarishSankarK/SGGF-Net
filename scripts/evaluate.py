"""
Evaluation script for SGGF-Net with visualization
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Colab/headless systems
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SGGFNet
from utils import VisDroneDataset, AITODDataset, HITUAVDataset, get_val_transform
from utils.metrics import calculate_map, calculate_ap50, calculate_precision_recall_f1, calculate_iou


def collate_fn(batch):
    """Custom collate function for batching"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def calculate_per_class_ap(predictions, targets, num_classes, iou_threshold=0.5):
    """Calculate AP for each class"""
    class_aps = {}
    
    for class_id in range(1, num_classes):  # Skip background
        # Collect predictions and targets for this class, grouped by image
        pred_boxes_by_img = []
        pred_scores_by_img = []
        gt_boxes_by_img = []
        
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Get predictions for this class
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                pred_boxes_by_img.append(pred['boxes'][class_mask])
                pred_scores_by_img.append(pred['scores'][class_mask])
            else:
                pred_boxes_by_img.append(torch.zeros(0, 4))
                pred_scores_by_img.append(torch.zeros(0))
            
            # Get ground truth for this class
            gt_mask = target['labels'] == class_id
            if gt_mask.any():
                gt_boxes_by_img.append(target['boxes'][gt_mask])
            else:
                gt_boxes_by_img.append(torch.zeros(0, 4))
        
        # Calculate AP for this class
        all_pred_boxes = []
        all_pred_scores = []
        pred_img_indices = []
        
        for img_idx, (pred_boxes, pred_scores) in enumerate(zip(pred_boxes_by_img, pred_scores_by_img)):
            if len(pred_boxes) > 0:
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                pred_img_indices.extend([img_idx] * len(pred_boxes))
        
        if len(all_pred_boxes) > 0:
            # Flatten predictions
            all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
            all_pred_scores = torch.cat(all_pred_scores, dim=0)
            
            # Sort by score
            sorted_indices = torch.argsort(all_pred_scores, descending=True)
            all_pred_boxes = all_pred_boxes[sorted_indices]
            all_pred_scores = all_pred_scores[sorted_indices]
            pred_img_indices = [pred_img_indices[i] for i in sorted_indices]
            
            # Match predictions to ground truth
            tp = torch.zeros(len(all_pred_boxes))
            fp = torch.zeros(len(all_pred_boxes))
            gt_matched = [torch.zeros(len(gt), dtype=torch.bool) for gt in gt_boxes_by_img]
            
            for i, (pred_box, img_idx) in enumerate(zip(all_pred_boxes, pred_img_indices)):
                img_gt_boxes = gt_boxes_by_img[img_idx]
                
                if len(img_gt_boxes) > 0:
                    ious = calculate_iou(pred_box.unsqueeze(0), img_gt_boxes)
                    max_iou, gt_idx = ious.max(dim=1)
                    max_iou_val = max_iou.item()
                    gt_idx_val = gt_idx.item()
                    
                    if max_iou_val >= iou_threshold and not gt_matched[img_idx][gt_idx_val]:
                        tp[i] = 1
                        gt_matched[img_idx][gt_idx_val] = True
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1
            
            # Calculate precision and recall
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            
            total_gt = sum(len(gt) for gt in gt_boxes_by_img)
            if total_gt > 0:
                recalls = (tp_cumsum / total_gt).numpy()
                precisions = (tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)).numpy()
                
                # Calculate AP
                recalls = np.concatenate(([0.0], recalls, [1.0]))
                precisions = np.concatenate(([0.0], precisions, [0.0]))
                
                for i in range(precisions.size - 1, 0, -1):
                    precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
                
                indices = np.where(recalls[1:] != recalls[:-1])[0]
                ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
                class_aps[class_id] = ap
            else:
                class_aps[class_id] = 0.0
        else:
            class_aps[class_id] = 0.0
    
    return class_aps


def create_visualizations(metrics, class_aps, output_dir, checkpoint_name):
    """Create visualization graphs for evaluation metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Class names for HIT-UAV (6 classes: background + 5)
    class_names = ['Background', 'Person', 'Car', 'Bicycle', 'Tricycle', 'Awning-tricycle']
    if len(class_aps) > 5:
        # If more classes, use generic names
        class_names = ['Background'] + [f'Class {i}' for i in range(1, len(class_aps) + 1)]
    
    # 1. Overall Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['mAP', 'AP50', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [
        metrics['mAP'],
        metrics['AP50'],
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1']
    ]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Overall Evaluation Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, max(metrics_values) * 1.2 if max(metrics_values) > 0 else 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved: {os.path.join(output_dir, "overall_metrics.png")}')
    
    # 2. Per-Class AP Bar Chart
    if class_aps:
        fig, ax = plt.subplots(figsize=(12, 6))
        class_ids = sorted(class_aps.keys())
        class_labels = [class_names[c] if c < len(class_names) else f'Class {c}' for c in class_ids]
        ap_values = [class_aps[c] for c in class_ids]
        
        bars = ax.bar(class_labels, ap_values, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, ap_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Average Precision (AP)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Average Precision (AP@0.5)', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0, max(ap_values) * 1.2 if max(ap_values) > 0 else 1.0])
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_ap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✓ Saved: {os.path.join(output_dir, "per_class_ap.png")}')
    
    # 3. Metrics Comparison (if multiple checkpoints exist, this is a placeholder)
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_to_plot = ['mAP', 'AP50', 'Precision', 'Recall', 'F1']
    values_to_plot = [metrics[m] for m in metrics_to_plot]
    
    # Create a pie chart for metric distribution
    colors_pie = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    wedges, texts, autotexts = ax.pie(values_to_plot, labels=metrics_to_plot, autopct='%1.3f',
                                      colors=colors_pie, startangle=90, textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Metrics Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved: {os.path.join(output_dir, "metrics_distribution.png")}')


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
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save visualization plots')
    
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
    
    # MPS: Disable num_workers (MPS has memory issues)
    num_workers = 0 if device.type == 'mps' else 2
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
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
                print(f'Processed {batch_idx+1}/{len(dataloader)} batches', flush=True)
    
    # Move predictions and targets to CPU for metric calculation
    print('\nMoving predictions to CPU for metric calculation...')
    cpu_predictions = []
    cpu_targets = []
    
    for pred, target in zip(predictions, targets_list):
        cpu_pred = {
            'boxes': pred['boxes'].cpu() if isinstance(pred['boxes'], torch.Tensor) else pred['boxes'],
            'labels': pred['labels'].cpu() if isinstance(pred['labels'], torch.Tensor) else pred['labels'],
            'scores': pred['scores'].cpu() if isinstance(pred['scores'], torch.Tensor) else pred['scores']
        }
        cpu_target = {
            'boxes': target['boxes'].cpu() if isinstance(target['boxes'], torch.Tensor) else target['boxes'],
            'labels': target['labels'].cpu() if isinstance(target['labels'], torch.Tensor) else target['labels']
        }
        cpu_predictions.append(cpu_pred)
        cpu_targets.append(cpu_target)
    
    # Calculate metrics
    print('\nCalculating metrics...')
    print('  This may take a few minutes for large datasets...', flush=True)
    mAP = calculate_map(cpu_predictions, cpu_targets, args.num_classes, verbose=True)
    print('\n  Calculating AP50...', flush=True)
    AP50 = calculate_ap50(cpu_predictions, cpu_targets, args.num_classes, verbose=True)
    print('\n  Calculating Precision/Recall/F1...', flush=True)
    precision, recall, f1 = calculate_precision_recall_f1(cpu_predictions, cpu_targets, args.num_classes)
    
    # Calculate per-class AP
    print('\n  Calculating per-class AP...', flush=True)
    class_aps = calculate_per_class_ap(cpu_predictions, cpu_targets, args.num_classes, iou_threshold=0.5)
    
    # Print results
    print('\n' + '='*70)
    print('Evaluation Results')
    print('='*70)
    print(f'mAP: {mAP:.4f}')
    print(f'AP50: {AP50:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('\nPer-Class AP@0.5:')
    for class_id, ap in sorted(class_aps.items()):
        print(f'  Class {class_id}: {ap:.4f}')
    print('='*70)
    
    # Create visualizations
    print('\nGenerating visualization graphs...', flush=True)
    metrics = {
        'mAP': mAP,
        'AP50': AP50,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    
    checkpoint_name = os.path.basename(args.checkpoint).replace('.pth', '')
    create_visualizations(metrics, class_aps, args.output_dir, checkpoint_name)
    
    print(f'\n✓ Evaluation completed!')
    print(f'✓ Visualizations saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()

