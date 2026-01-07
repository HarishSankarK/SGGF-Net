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
        boxes1: (N, 4) tensor in format [x_center, y_center, width, height]
        boxes2: (M, 4) tensor in format [x_center, y_center, width, height]
    Returns:
        IoU matrix of shape (N, M)
    """
    # Convert to [x1, y1, x2, y2] format
    def to_corners(boxes):
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    boxes1_corners = to_corners(boxes1)
    boxes2_corners = to_corners(boxes2)
    
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


def calculate_map(predictions, targets, num_classes, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of dicts with 'boxes', 'labels', 'scores'
        targets: List of dicts with 'boxes', 'labels'
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds for AP calculation
    Returns:
        mAP value
    """
    aps = []
    
    for iou_thresh in iou_thresholds:
        class_aps = []
        
        for class_id in range(1, num_classes):  # Skip background
            # Collect predictions and targets for this class
            pred_boxes = []
            pred_scores = []
            gt_boxes = []
            
            for pred, target in zip(predictions, targets):
                # Get predictions for this class
                class_mask = pred['labels'] == class_id
                if class_mask.any():
                    pred_boxes.append(pred['boxes'][class_mask])
                    pred_scores.append(pred['scores'][class_mask])
                else:
                    pred_boxes.append(torch.zeros(0, 4))
                    pred_scores.append(torch.zeros(0))
                
                # Get ground truth for this class
                gt_mask = target['labels'] == class_id
                if gt_mask.any():
                    gt_boxes.append(target['boxes'][gt_mask])
                else:
                    gt_boxes.append(torch.zeros(0, 4))
            
            # Calculate AP for this class
            if len(pred_boxes) > 0:
                # Flatten predictions
                all_pred_boxes = torch.cat(pred_boxes, dim=0)
                all_pred_scores = torch.cat(pred_scores, dim=0)
                
                # Sort by score
                sorted_indices = torch.argsort(all_pred_scores, descending=True)
                all_pred_boxes = all_pred_boxes[sorted_indices]
                all_pred_scores = all_pred_scores[sorted_indices]
                
                # Match predictions to ground truth
                tp = torch.zeros(len(all_pred_boxes))
                fp = torch.zeros(len(all_pred_boxes))
                
                gt_matched = [torch.zeros(len(gt), dtype=torch.bool) for gt in gt_boxes]
                
                for i, pred_box in enumerate(all_pred_boxes):
                    # Find best matching GT
                    best_iou = 0
                    best_gt_idx = -1
                    best_img_idx = -1
                    
                    for img_idx, img_gt_boxes in enumerate(gt_boxes):
                        if len(img_gt_boxes) > 0:
                            ious = calculate_iou(pred_box.unsqueeze(0), img_gt_boxes)
                            max_iou, gt_idx = ious.max(dim=1)
                            if max_iou.item() > best_iou:
                                best_iou = max_iou.item()
                                best_gt_idx = gt_idx.item()
                                best_img_idx = img_idx
                    
                    if best_iou >= iou_thresh and not gt_matched[best_img_idx][best_gt_idx]:
                        tp[i] = 1
                        gt_matched[best_img_idx][best_gt_idx] = True
                    else:
                        fp[i] = 1
                
                # Calculate precision and recall
                tp_cumsum = torch.cumsum(tp, dim=0)
                fp_cumsum = torch.cumsum(fp, dim=0)
                
                total_gt = sum(len(gt) for gt in gt_boxes)
                recalls = (tp_cumsum / total_gt).numpy() if total_gt > 0 else np.zeros(len(tp))
                precisions = (tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)).numpy()
                
                ap = calculate_ap(recalls, precisions)
                class_aps.append(ap)
        
        # Average AP across classes
        if class_aps:
            aps.append(np.mean(class_aps))
        else:
            aps.append(0.0)
    
    # mAP is average across all IoU thresholds
    return np.mean(aps)


def calculate_ap50(predictions, targets, num_classes):
    """Calculate AP at IoU threshold 0.5"""
    return calculate_map(predictions, targets, num_classes, iou_thresholds=[0.5])


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

