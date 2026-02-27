"""
Generate evaluation figures for manuscript: PR curves, confusion matrix, mAP vs IoU,
metrics bar chart, detection examples, and training curves.
Usage:
  python scripts/generate_eval_figures.py --dataset hituav --data_dir data/hit-uav \
    --checkpoint checkpoints/best.pth --output_dir ../Paper/figures
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as F_tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import FusionYOLOv11
from utils.dataset import DroneRGBTDataset, HITUAVDataset
from utils.transforms import get_val_transform
from utils.metrics import (
    calculate_map, calculate_ap, calculate_ap50, calculate_precision_recall_f1,
    calculate_iou, get_precision_recall_curves
)

# Class names (1=person, 2=vehicle)
CLASS_NAMES = {1: 'Person', 2: 'Vehicle'}
COLORS = {1: (0, 255, 0), 2: (0, 0, 255)}  # Green, Blue for drawing


def collate_fn_single(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def run_evaluation(model, dataloader, device, num_classes, conf_threshold=0.25):
    """Run model and return predictions + targets."""
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Evaluating'):
            images, targets = batch_data
            images = [img.to(device) for img in images]
            preds = model.predict(images, images, conf_threshold=conf_threshold, nms_threshold=0.5)
            all_predictions.extend(preds)
            all_targets.extend(targets)
    
    formatted_targets = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in all_targets]
    formatted_predictions = [
        {'boxes': p['boxes'].cpu(), 'scores': p['scores'].cpu(), 'labels': p['labels'].cpu()}
        for p in all_predictions
    ]
    return formatted_predictions, formatted_targets, all_predictions, all_targets


def build_confusion_matrix(predictions, targets, num_classes, iou_threshold=0.5):
    """Build confusion matrix: rows=predicted, cols=ground truth. Uses 0 for background/unmatched."""
    n = num_classes  # 0=bg, 1=person, 2=vehicle
    cm = np.zeros((n, n))
    
    for pred, target in zip(predictions, targets):
        pred_boxes, pred_labels = pred['boxes'], pred['labels']
        gt_boxes, gt_labels = target['boxes'], target['labels']
        
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        if len(gt_boxes) == 0:
            for pl in pred_labels:
                cm[pl.item(), 0] += 1  # FP: predicted class, no GT
            continue
        if len(pred_boxes) == 0:
            for gl in gt_labels:
                cm[0, gl.item()] += 1  # FN: no pred, GT class
            continue
        
        ious = calculate_iou(pred_boxes, gt_boxes)
        pred_matched = torch.zeros(len(pred_boxes), dtype=torch.bool)
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        for _ in range(min(len(pred_boxes), len(gt_boxes))):
            max_iou, pi, gi = -1, -1, -1
            for i in range(len(pred_boxes)):
                if pred_matched[i]:
                    continue
                for j in range(len(gt_boxes)):
                    if gt_matched[j]:
                        continue
                    if ious[i, j].item() > max_iou:
                        max_iou = ious[i, j].item()
                        pi, gi = i, j
            if max_iou < iou_threshold or pi < 0:
                break
            pred_matched[pi] = True
            gt_matched[gi] = True
            pl, gl = pred_labels[pi].item(), gt_labels[gi].item()
            cm[pl, gl] += 1
        
        for i in range(len(pred_boxes)):
            if not pred_matched[i]:
                cm[pred_labels[i].item(), 0] += 1
        for j in range(len(gt_boxes)):
            if not gt_matched[j]:
                cm[0, gt_labels[j].item()] += 1
    
    return cm


def get_per_class_metrics(predictions, targets, num_classes, curves, iou_threshold=0.5):
    """Precision, Recall, F1, AP50 per class."""
    results = {}
    for c in range(1, num_classes):
        tp = fp = fn = 0
        pred_boxes_all, pred_scores_all, pred_img_idx = [], [], []
        gt_boxes_by_img = []
        
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            cm = pred['labels'] == c
            pb = pred['boxes'][cm] if cm.any() else torch.zeros(0, 4)
            ps = pred['scores'][cm] if cm.any() else torch.zeros(0)
            gm = target['labels'] == c
            gb = target['boxes'][gm] if gm.any() else torch.zeros(0, 4)
            gt_boxes_by_img.append(gb)
            if len(pb) > 0:
                pred_boxes_all.append(pb)
                pred_scores_all.append(ps)
                pred_img_idx.extend([img_idx] * len(pb))
        
        if len(pred_boxes_all) == 0:
            total_gt = sum(len(g) for g in gt_boxes_by_img)
            results[c] = {'precision': 0, 'recall': 0, 'f1': 0, 'ap50': 0}
            continue
        
        pred_boxes_all = torch.cat(pred_boxes_all)
        pred_scores_all = torch.cat(pred_scores_all)
        sidx = torch.argsort(pred_scores_all, descending=True)
        pred_boxes_all = pred_boxes_all[sidx]
        pred_img_idx = [pred_img_idx[i] for i in sidx]
        
        gt_matched = [torch.zeros(len(g), dtype=torch.bool) for g in gt_boxes_by_img]
        total_gt = sum(len(g) for g in gt_boxes_by_img)
        tp_count = 0
        
        for i, img_idx in enumerate(pred_img_idx):
            img_gt = gt_boxes_by_img[img_idx]
            if len(img_gt) == 0:
                continue
            ious = calculate_iou(pred_boxes_all[i:i+1], img_gt)
            max_iou, gi = ious.max(dim=1)
            if max_iou.item() >= iou_threshold and not gt_matched[img_idx][gi.item()]:
                tp_count += 1
                gt_matched[img_idx][gi.item()] = True
        
        fp_count = len(pred_boxes_all) - tp_count
        fn_count = total_gt - tp_count
        prec = tp_count / (tp_count + fp_count + 1e-6)
        rec = tp_count / (tp_count + fn_count + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        # AP50 from PR curve (curves computed at IoU=0.5)
        ap50 = 0.0
        if c in curves:
            prec_arr, rec_arr = curves[c]
            if len(prec_arr) >= 2:
                ap50 = calculate_ap(rec_arr, prec_arr)
        results[c] = {'precision': prec, 'recall': rec, 'f1': f1, 'ap50': ap50}
    
    return results


def plot_pr_curves(curves, num_classes, output_path):
    """Plot Precision-Recall curves per class."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for c in range(1, num_classes):
        if c not in curves:
            continue
        prec, rec = curves[c]
        if len(prec) < 2:
            continue
        ax.plot(rec, prec, label=CLASS_NAMES.get(c, f'Class {c}'), linewidth=2)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves (IoU=0.5)', fontsize=12)
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_confusion_matrix(cm, num_classes, output_path):
    """Plot confusion matrix heatmap."""
    # Use classes 1,2 for display; row/col 0 = background/unmatched
    labels = ['BG'] + [CLASS_NAMES.get(i, f'C{i}') for i in range(1, num_classes)]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted')
    for i in range(num_classes):
        for j in range(num_classes):
            v = cm[i, j]
            ax.text(j, i, f'{int(v)}', ha='center', va='center', color='black' if v < cm.max()/2 else 'white')
    plt.colorbar(im, ax=ax, label='Count')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_map_vs_iou(predictions, targets, num_classes, output_path):
    """Plot mAP vs IoU threshold."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps = []
    for iou in iou_thresholds:
        m = calculate_map(predictions, targets, num_classes, iou_thresholds=[float(iou)], verbose=False)
        maps.append(m)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(iou_thresholds, maps, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('IoU Threshold', fontsize=11)
    ax.set_ylabel('mAP', fontsize=11)
    ax.set_title('mAP vs IoU Threshold', fontsize=12)
    ax.set_xlim([0.45, 1.0])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_metrics_bar(per_class_metrics, output_path):
    """Bar chart of Precision, Recall, F1, AP50 per class."""
    classes = list(per_class_metrics.keys())
    names = [CLASS_NAMES.get(c, f'C{c}') for c in classes]
    metrics = ['precision', 'recall', 'f1', 'ap50']
    labels = ['Precision', 'Recall', 'F1', 'AP50']
    data = np.array([[per_class_metrics[c][m] for m in metrics] for c in classes])
    
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (m, lb) in enumerate(zip(metrics, labels)):
        ax.bar(x + i * width, data[:, i], width, label=lb)
    ax.set_ylabel('Score')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.set_title('Per-Class Metrics (IoU=0.5)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def save_detection_examples(model, dataset, device, num_classes, output_path, num_examples=6, conf_threshold=0.25):
    """Save sample images with predicted boxes overlaid."""
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, num_examples, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for ax_idx, idx in enumerate(indices):
        img_tensor, target = dataset[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        H, W = img_np.shape[:2]
        
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            preds = model.predict(inp, inp, conf_threshold=conf_threshold, nms_threshold=0.5)[0]
        
        ax = axes[ax_idx]
        ax.imshow(img_np)
        
        for i in range(len(preds['boxes'])):
            x1, y1, x2, y2 = preds['boxes'][i].cpu().tolist()
            score = preds['scores'][i].item()
            lab = preds['labels'][i].item()
            color = ['green', 'blue'][lab - 1] if lab in (1, 2) else 'red'
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f'{CLASS_NAMES.get(lab, lab)} {score:.2f}', color='white',
                    fontsize=8, bbox=dict(facecolor=color, alpha=0.7))
        
        ax.axis('off')
        ax.set_title(f'Image {idx}')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def plot_training_curves(log_path, output_path):
    """Plot loss and mAP from training log. Supports CSV: epoch,train_loss,val_map,val_ap50,val_ap25"""
    if not log_path or not os.path.exists(log_path):
        print(f'  Skipped training curves (no log: {log_path})')
        return
    epochs, losses, maps, ap50s = [], [], [], []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('epoch'):
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    epochs.append(int(parts[0]))
                    losses.append(float(parts[1]))
                    maps.append(float(parts[2]))
                    ap50s.append(float(parts[3]))
                except (ValueError, IndexError):
                    continue
    if not epochs:
        print('  Skipped training curves (could not parse log)')
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs[:len(losses)], losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if maps:
        ax2.plot(epochs[:len(maps)], maps, 'g-', label='Val mAP')
    if ap50s:
        ax2.plot(epochs[:len(ap50s)], ap50s, 'r-', label='Val AP50')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation figures for manuscript')
    parser.add_argument('--dataset', type=str, default='hituav', choices=['hituav', 'dronergbt'])
    parser.add_argument('--data_dir', type=str, default='data/hit-uav')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../Paper/figures')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--conf_threshold', type=float, default=0.25)
    parser.add_argument('--training_log', type=str, default='', help='Path to training_log.csv (auto-detected from checkpoint dir if not set)')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.training_log and args.checkpoint:
        ckpt_dir = os.path.dirname(args.checkpoint)
        default_log = os.path.join(ckpt_dir, 'training_log.csv')
        if os.path.exists(default_log):
            args.training_log = default_log
    
    print('Loading model and dataset...')
    transform = get_val_transform(max_size=640)
    if args.dataset == 'hituav':
        dataset = HITUAVDataset(root_dir=args.data_dir, split=args.split, transform=transform, use_person_vehicle=True)
        collate_fn = collate_fn_single
    elif args.dataset == 'dronergbt':
        dataset = DroneRGBTDataset(root_dir=args.data_dir, split=args.split, transform=transform)
        def _collate_dronergbt(batch):
            images = [item[0][0] for item in batch]
            targets = [item[1] for item in batch]
            return images, targets
        collate_fn = _collate_dronergbt
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Use hituav or dronergbt.")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone = ckpt.get('backbone', 'resnet50') if isinstance(ckpt, dict) else 'resnet50'
    model_num_classes = ckpt.get('num_classes') if isinstance(ckpt, dict) else None
    if model_num_classes is None:
        model_num_classes = 2 if args.num_classes >= 3 else args.num_classes
    model = FusionYOLOv11(num_classes=model_num_classes, pretrained=False, fusion_type='concat_attention', backbone=backbone)
    model.load_state_dict(ckpt.get('ema_state_dict', ckpt['model_state_dict']))
    model = model.to(device).eval()
    
    print('Running evaluation...')
    predictions, targets, _, _ = run_evaluation(model, loader, device, args.num_classes, args.conf_threshold)
    
    print('Generating figures...')
    
    curves = get_precision_recall_curves(predictions, targets, args.num_classes, iou_threshold=0.5)
    plot_pr_curves(curves, args.num_classes, os.path.join(args.output_dir, 'pr_curve.pdf'))
    
    cm = build_confusion_matrix(predictions, targets, args.num_classes, iou_threshold=0.5)
    plot_confusion_matrix(cm, args.num_classes, os.path.join(args.output_dir, 'confusion_matrix.pdf'))
    
    plot_map_vs_iou(predictions, targets, args.num_classes, os.path.join(args.output_dir, 'map_vs_iou.pdf'))
    
    per_class = get_per_class_metrics(predictions, targets, args.num_classes, curves, iou_threshold=0.5)
    plot_metrics_bar(per_class, os.path.join(args.output_dir, 'metrics_bar.pdf'))
    
    save_detection_examples(model, dataset, device, args.num_classes,
                            os.path.join(args.output_dir, 'detection_examples.pdf'), num_examples=6,
                            conf_threshold=args.conf_threshold)
    
    plot_training_curves(args.training_log, os.path.join(args.output_dir, 'training_curves.pdf'))
    
    print('Done. Figures saved to', args.output_dir)


if __name__ == '__main__':
    main()
