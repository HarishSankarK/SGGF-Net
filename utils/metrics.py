"""
Evaluation metrics: mAP, AP50, Precision, Recall, F1
"""

import torch
import numpy as np
from collections import defaultdict


def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: (N, 4) tensor in format [x1, y1, x2, y2]
        boxes2: (M, 4) tensor in format [x1, y1, x2, y2]
    Returns:
        IoU matrix of shape (N, M)
    """
    # Ensure both tensors are on the same device
    if boxes1.device != boxes2.device:
        boxes2 = boxes2.to(boxes1.device)
    
    boxes1_corners = boxes1
    boxes2_corners = boxes2
    
    # Calculate intersection
    x1_max = torch.max(boxes1_corners[:, 0:1], boxes2_corners[:, 0].unsqueeze(0))
    y1_max = torch.max(boxes1_corners[:, 1:2], boxes2_corners[:, 1].unsqueeze(0))
    x2_min = torch.min(boxes1_corners[:, 2:3], boxes2_corners[:, 2].unsqueeze(0))
    y2_min = torch.min(boxes1_corners[:, 3:4], boxes2_corners[:, 3].unsqueeze(0))
    
    intersection = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(y2_min - y1_max, min=0)
    
    # Calculate areas
    area1 = (boxes1_corners[:, 2] - boxes1_corners[:, 0]) * (boxes1_corners[:, 3] - boxes1_corners[:, 1])
    area2 = (boxes2_corners[:, 2] - boxes2_corners[:, 0]) * (boxes2_corners[:, 3] - boxes2_corners[:, 1])
    
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


def calculate_ap(recalls, precisions):
    """Calculate Average Precision from precision-recall curve"""
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def calculate_map(predictions, targets, num_classes, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], verbose=False):
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of dicts with 'boxes', 'labels', 'scores'
        targets: List of dicts with 'boxes', 'labels'
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds for AP calculation
        verbose: Print progress messages
    Returns:
        mAP value
    """
    aps = []
    num_thresholds = len(iou_thresholds)
    num_class_loops = num_classes - 1  # Skip background
    
    for thresh_idx, iou_thresh in enumerate(iou_thresholds):
        if verbose:
            print(f'  Calculating AP at IoU={iou_thresh:.2f} ({thresh_idx+1}/{num_thresholds})...')
        
        class_aps = []
        
        for class_id in range(1, num_classes):  # Skip background
            if verbose and num_class_loops > 3:
                print(f'    Class {class_id}/{num_classes-1}...', end=' ', flush=True)
            
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
            # Collect all predictions with their image indices
            all_pred_boxes = []
            all_pred_scores = []
            pred_img_indices = []  # Track which image each prediction belongs to
            
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
                
                # Match predictions to ground truth (WITHIN THE SAME IMAGE)
                tp = torch.zeros(len(all_pred_boxes))
                fp = torch.zeros(len(all_pred_boxes))
                
                gt_matched = [torch.zeros(len(gt), dtype=torch.bool) for gt in gt_boxes_by_img]
                
                # Process predictions in batches for progress tracking
                total_preds = len(all_pred_boxes)
                batch_size_progress = max(1, total_preds // 10) if verbose and total_preds > 100 else total_preds
                
                for i, (pred_box, img_idx) in enumerate(zip(all_pred_boxes, pred_img_indices)):
                    # Only match to GT in the same image
                    img_gt_boxes = gt_boxes_by_img[img_idx]
                    
                    if len(img_gt_boxes) > 0:
                        # Calculate IoU with GT boxes in this image only
                        ious = calculate_iou(pred_box.unsqueeze(0), img_gt_boxes)
                        max_iou, gt_idx = ious.max(dim=1)
                        max_iou_val = max_iou.item()
                        gt_idx_val = gt_idx.item()
                        
                        # Check if this GT box is already matched
                        if max_iou_val >= iou_thresh and not gt_matched[img_idx][gt_idx_val]:
                            tp[i] = 1
                            gt_matched[img_idx][gt_idx_val] = True
                        else:
                            fp[i] = 1
                    else:
                        # No GT boxes in this image -> FP
                        fp[i] = 1
                    
                    # Progress update
                    if verbose and (i + 1) % batch_size_progress == 0:
                        print(f'{i+1}/{total_preds}', end=' ', flush=True)
                
                if verbose and num_class_loops > 3:
                    print()  # New line after progress
                
                # Calculate precision and recall
                tp_cumsum = torch.cumsum(tp, dim=0)
                fp_cumsum = torch.cumsum(fp, dim=0)
                
                total_gt = sum(len(gt) for gt in gt_boxes_by_img)
                recalls = (tp_cumsum / total_gt).numpy() if total_gt > 0 else np.zeros(len(tp))
                precisions = (tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)).numpy()
                
                ap = calculate_ap(recalls, precisions)
                class_aps.append(ap)
            else:
                # No predictions for this class — AP=0 if GT exists
                total_gt = sum(len(gt) for gt in gt_boxes_by_img)
                if total_gt > 0:
                    class_aps.append(0.0)
        
        # Average AP across classes
        if class_aps:
            aps.append(np.mean(class_aps))
        else:
            aps.append(0.0)
    
    # mAP is average across all IoU thresholds
    return np.mean(aps)


def calculate_ap50(predictions, targets, num_classes, verbose=False):
    """Calculate AP at IoU threshold 0.5"""
    return calculate_map(predictions, targets, num_classes, iou_thresholds=[0.5], verbose=verbose)


def calculate_precision_recall_f1(predictions, targets, num_classes, iou_threshold=0.5):
    """
    Calculate Precision, Recall, and F1 score
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
    Returns:
        precision, recall, f1
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, target in zip(predictions, targets):
        # Match predictions to ground truth
        if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
            ious = calculate_iou(pred['boxes'], target['boxes'])
            max_ious, _ = ious.max(dim=1)
            
            # TP: IoU >= threshold
            tp = (max_ious >= iou_threshold).sum().item()
            total_tp += tp
            total_fp += len(pred['boxes']) - tp
        else:
            if len(pred['boxes']) > 0:
                total_fp += len(pred['boxes'])
        
        # FN: unmatched ground truth
        if len(target['boxes']) > 0:
            if len(pred['boxes']) > 0:
                ious = calculate_iou(pred['boxes'], target['boxes'])
                matched_gt = (ious.max(dim=0)[0] >= iou_threshold).sum().item()
                total_fn += len(target['boxes']) - matched_gt
            else:
                total_fn += len(target['boxes'])
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, f1
