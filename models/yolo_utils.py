"""
YOLOv11 Utilities
Helper functions for anchor-free detection, post-processing, etc.
"""

import torch
import torch.nn.functional as F
import math


def generate_grid_points(feature_map_size, stride, device=None):
    """
    Generate grid points for anchor-free detection
    
    Args:
        feature_map_size: (H, W) of feature map
        stride: Stride of feature map relative to input image
        device: Device to create tensors on
    Returns:
        grid_points: (H, W, 2) tensor with (x, y) coordinates in image space
    """
    H, W = feature_map_size
    y_coords = torch.arange(0.5, H, 1.0, device=device) * stride
    x_coords = torch.arange(0.5, W, 1.0, device=device) * stride
    
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_points = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
    
    return grid_points


def decode_bbox(predictions, grid_points, stride):
    """
    Decode YOLOv11 predictions to absolute bounding boxes
    
    Args:
        predictions: (B, H, W, 4) or (N, 4) bbox predictions in (dx, dy, dw, dh) format
        grid_points: (H, W, 2) or (N, 2) grid point coordinates
        stride: Stride of feature map
    Returns:
        boxes: (B, H, W, 4) or (N, 4) boxes in (x_center, y_center, w, h) format
    """
    if predictions.dim() == 4:
        B, H, W, _ = predictions.shape
        if grid_points.dim() == 4:
            grid_points = grid_points.squeeze(0)
        grid = grid_points.unsqueeze(0).expand(B, H, W, 2)

        dx, dy, dw, dh = predictions[..., 0], predictions[..., 1], predictions[..., 2], predictions[..., 3]

        # Centered sigmoid: range (-0.5*stride, 1.5*stride) — allows offsets in both directions
        dx = (2.0 * torch.sigmoid(dx) - 0.5) * stride
        dy = (2.0 * torch.sigmoid(dy) - 0.5) * stride
        dw = torch.exp(torch.clamp(dw, max=10)) * stride
        dh = torch.exp(torch.clamp(dh, max=10)) * stride

        x_center = grid[..., 0] + dx
        y_center = grid[..., 1] + dy

        boxes = torch.stack([x_center, y_center, dw, dh], dim=-1)
        return boxes
    else:
        N = predictions.shape[0]
        if grid_points.dim() == 2 and grid_points.shape[0] == N:
            grid = grid_points
        else:
            raise ValueError(f"grid_points shape {grid_points.shape} doesn't match predictions shape {predictions.shape}")

        dx, dy, dw, dh = predictions[..., 0], predictions[..., 1], predictions[..., 2], predictions[..., 3]
        dx = (2.0 * torch.sigmoid(dx) - 0.5) * stride
        dy = (2.0 * torch.sigmoid(dy) - 0.5) * stride
        dw = torch.exp(torch.clamp(dw, max=10)) * stride
        dh = torch.exp(torch.clamp(dh, max=10)) * stride

        x_center = grid[..., 0] + dx
        y_center = grid[..., 1] + dy
        boxes = torch.stack([x_center, y_center, dw, dh], dim=-1)
        return boxes


def compute_iou(boxes1, boxes2, chunk_size=8192):
    """
    Compute IoU between two sets of boxes in center format.
    
    Args:
        boxes1: (N, 4) in (x_center, y_center, w, h) format
        boxes2: (M, 4) in (x_center, y_center, w, h) format
        chunk_size: Process boxes1 in chunks to limit memory
    Returns:
        iou: (N, M) IoU matrix
    """
    def to_corners(boxes):
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=1)
    
    N, M = boxes1.shape[0], boxes2.shape[0]
    if N <= chunk_size:
        b1 = to_corners(boxes1)
        b2 = to_corners(boxes2)
        b1e = b1.unsqueeze(1)
        b2e = b2.unsqueeze(0)
        ix1 = torch.max(b1e[..., 0], b2e[..., 0])
        iy1 = torch.max(b1e[..., 1], b2e[..., 1])
        ix2 = torch.min(b1e[..., 2], b2e[..., 2])
        iy2 = torch.min(b1e[..., 3], b2e[..., 3])
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
        a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = a1.unsqueeze(1) + a2.unsqueeze(0) - inter
        return inter / (union + 1e-7)
    
    iou_list = []
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = compute_iou(boxes1[i:end], boxes2, chunk_size=chunk_size * 2)
        iou_list.append(chunk)
    return torch.cat(iou_list, dim=0)


def assign_targets_to_predictions(predictions_list, grid_points_list, strides, targets, image_size,
                                  pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign GT targets to anchor-free predictions using center-based assignment.
    
    For each GT box, the grid cell whose center falls inside the GT box is positive.
    Additionally, the top-k closest grid cells to each GT center are considered candidates.
    This ensures positive samples even when IoU is low (early training).
    
    Args:
        predictions_list: List of bbox predictions for each scale [(B, H_i, W_i, 4), ...]
        grid_points_list: List of grid points for each scale [(H_i, W_i, 2), ...]
        strides: List of strides for each scale
        targets: List of target dicts with 'boxes' (xyxy) and 'labels'
        image_size: (H, W) of input image
    Returns:
        assigned_targets: List of dicts with 'labels', 'bbox_targets', 'obj_targets', 'cls_targets'
    """
    B = len(targets)
    num_scales = len(predictions_list)
    
    assigned_targets = []
    
    for scale_idx in range(num_scales):
        predictions = predictions_list[scale_idx]  # (B, H, W, 4)
        grid_points = grid_points_list[scale_idx]  # (H, W, 2)
        stride = strides[scale_idx]
        
        B, H, W, _ = predictions.shape
        
        # Initialize targets
        labels = torch.zeros(B, H, W, dtype=torch.long, device=predictions.device)
        bbox_targets = torch.zeros(B, H, W, 4, dtype=torch.float32, device=predictions.device)
        obj_targets = torch.zeros(B, H, W, dtype=torch.float32, device=predictions.device)
        cls_targets = torch.zeros(B, H, W, dtype=torch.long, device=predictions.device)
        
        # Grid points flat: (H*W, 2)
        grid_flat = grid_points.reshape(-1, 2)
        
        max_gt_boxes = 100
        for b in range(B):
            target = targets[b]
            gt_boxes = target['boxes']  # (M, 4) xyxy
            gt_labels = target['labels']  # (M,)
            if len(gt_boxes) > max_gt_boxes:
                perm = torch.randperm(len(gt_boxes), device=gt_boxes.device)[:max_gt_boxes]
                gt_boxes = gt_boxes[perm]
                gt_labels = gt_labels[perm]
            if len(gt_boxes) == 0:
                continue
            
            # GT center and size
            x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
            gt_cx = (x1 + x2) / 2  # (M,)
            gt_cy = (y1 + y2) / 2
            gt_w = (x2 - x1).clamp(min=1)
            gt_h = (y2 - y1).clamp(min=1)
            gt_boxes_center = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)  # (M, 4)
            
            # ---- Center-based assignment ----
            # For each GT box, find grid cells whose center falls inside the GT box
            # grid_flat: (H*W, 2), gt_boxes: (M, 4)
            gx = grid_flat[:, 0].unsqueeze(1)  # (H*W, 1)
            gy = grid_flat[:, 1].unsqueeze(1)
            inside = (gx >= x1.unsqueeze(0)) & (gx <= x2.unsqueeze(0)) & \
                     (gy >= y1.unsqueeze(0)) & (gy <= y2.unsqueeze(0))  # (H*W, M)
            
            # Also add top-k closest grid cells per GT (k=9) as candidates
            dist = torch.sqrt((gx - gt_cx.unsqueeze(0))**2 + (gy - gt_cy.unsqueeze(0))**2)  # (H*W, M)
            topk_k = min(9, H * W)
            _, topk_idx = dist.topk(topk_k, dim=0, largest=False)  # (k, M)
            topk_mask = torch.zeros_like(inside)
            for m in range(gt_boxes.shape[0]):
                topk_mask[topk_idx[:, m], m] = True
            
            # Positive candidates: inside OR top-k closest
            candidates = inside | topk_mask  # (H*W, M)
            
            # For each positive candidate, assign the closest GT
            # Set distance to inf for non-candidates
            dist_masked = dist.clone()
            dist_masked[~candidates] = float('inf')
            
            # For each grid cell, find the best (closest) GT
            min_dist, best_gt = dist_masked.min(dim=1)  # (H*W,)
            pos_mask = min_dist < float('inf')  # (H*W,)
            
            if not pos_mask.any():
                continue
            
            # Assign labels and targets
            pos_gt_idx = best_gt[pos_mask]
            matched_gt = gt_boxes_center[pos_gt_idx]  # (N_pos, 4) center format
            matched_labels = gt_labels[pos_gt_idx]  # (N_pos,)
            
            labels_flat = labels[b].reshape(-1)
            obj_flat = obj_targets[b].reshape(-1)
            cls_flat = cls_targets[b].reshape(-1)
            bbox_flat = bbox_targets[b].reshape(-1, 4)
            
            # Map class labels: labels from dataset are 1-indexed (1=person, 2=vehicle)
            # Head outputs num_classes channels (0-indexed). Map: label 1 → index 0, label 2 → index 1
            cls_indices = (matched_labels - 1).clamp(min=0)
            
            labels_flat[pos_mask] = matched_labels
            obj_flat[pos_mask] = 1.0
            cls_flat[pos_mask] = cls_indices
            
            # Compute bbox regression targets as deltas from grid cell
            pos_grid = grid_flat[pos_mask]  # (N_pos, 2)
            dx = (matched_gt[:, 0] - pos_grid[:, 0]) / stride
            dy = (matched_gt[:, 1] - pos_grid[:, 1]) / stride
            dw = torch.log(matched_gt[:, 2] / stride + 1e-6)
            dh = torch.log(matched_gt[:, 3] / stride + 1e-6)
            bbox_flat[pos_mask] = torch.stack([dx, dy, dw, dh], dim=1)
            
            labels[b] = labels_flat.reshape(H, W)
            obj_targets[b] = obj_flat.reshape(H, W)
            cls_targets[b] = cls_flat.reshape(H, W)
            bbox_targets[b] = bbox_flat.reshape(H, W, 4)
        
        assigned_targets.append({
            'labels': labels,
            'bbox_targets': bbox_targets,
            'obj_targets': obj_targets,
            'cls_targets': cls_targets
        })
    
    return assigned_targets


def post_process_predictions(predictions, grid_points_list, strides, conf_threshold=0.5, 
                             nms_threshold=0.5, max_detections=100):
    """
    Post-process YOLOv11 predictions: decode, filter, NMS
    
    Args:
        predictions: Dict with 'cls', 'obj', 'bbox' predictions
        grid_points_list: List of grid points for each scale
        strides: List of strides for each scale
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
        max_detections: Maximum detections per image
    Returns:
        results: List of detection dicts, each with 'boxes' (xyxy), 'scores', 'labels' (1-indexed)
    """
    cls_preds = predictions['cls']
    obj_preds = predictions['obj']
    bbox_preds = predictions['bbox']
    
    B = cls_preds[0].shape[0]
    num_scales = len(cls_preds)
    num_classes = cls_preds[0].shape[-1]
    
    results = []
    
    for b in range(B):
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for scale_idx in range(num_scales):
            cls_pred = cls_preds[scale_idx][b]  # (num_anchors, H, W, num_classes)
            obj_pred = obj_preds[scale_idx][b]  # (num_anchors, H, W, 1)
            bbox_pred = bbox_preds[scale_idx][b]  # (num_anchors, H, W, 4)
            grid_points = grid_points_list[scale_idx]  # (H, W, 2)
            stride = strides[scale_idx]
            
            num_anchors_local = cls_pred.shape[0]
            H, W = cls_pred.shape[1], cls_pred.shape[2]
            
            cls_pred = cls_pred[0]  # (H, W, num_classes)
            obj_pred = obj_pred[0, ..., 0]  # (H, W)
            bbox_pred = bbox_pred[0]  # (H, W, 4)
            
            cls_prob = torch.sigmoid(cls_pred)  # (H, W, num_classes)
            obj_prob = torch.sigmoid(obj_pred)  # (H, W)
            
            conf = obj_prob.unsqueeze(-1) * cls_prob  # (H, W, num_classes)
            
            boxes = decode_bbox(bbox_pred.unsqueeze(0), grid_points, stride)[0]  # (H, W, 4) center format
            
            mask = conf > conf_threshold
            
            if mask.any():
                y_indices, x_indices, label_indices = torch.where(mask)
                
                boxes_selected = boxes[y_indices, x_indices]  # (N, 4) center format
                scores_selected = conf[y_indices, x_indices, label_indices]
                # label_indices are 0-indexed from head; convert to 1-indexed for output
                labels_selected = label_indices + 1
                
                all_boxes.append(boxes_selected)
                all_scores.append(scores_selected)
                all_labels.append(labels_selected)
        
        if len(all_boxes) > 0:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            from .anchor_utils import nms
            
            final_boxes = []
            final_scores = []
            final_labels = []
            
            unique_labels = all_labels.unique()
            for cls_id in unique_labels:
                cls_mask = all_labels == cls_id
                if not cls_mask.any():
                    continue
                
                cls_boxes = all_boxes[cls_mask]  # center format
                cls_scores = all_scores[cls_mask]
                
                keep_indices = nms(cls_boxes, cls_scores, iou_threshold=nms_threshold, 
                                  max_detections=max_detections)
                
                if len(keep_indices) > 0:
                    final_boxes.append(cls_boxes[keep_indices])
                    final_scores.append(cls_scores[keep_indices])
                    final_labels.append(torch.full((len(keep_indices),), cls_id, 
                                                   dtype=torch.long, device=all_boxes.device))
            
            if len(final_boxes) > 0:
                final_boxes = torch.cat(final_boxes, dim=0)
                final_scores = torch.cat(final_scores, dim=0)
                final_labels = torch.cat(final_labels, dim=0)
                
                # Convert to xyxy for output
                cx, cy, w, h = final_boxes[:, 0], final_boxes[:, 1], final_boxes[:, 2], final_boxes[:, 3]
                boxes_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)
                
                results.append({
                    'boxes': boxes_xyxy,
                    'scores': final_scores,
                    'labels': final_labels
                })
            else:
                results.append({
                    'boxes': torch.zeros((0, 4), dtype=torch.float32, device=all_boxes.device),
                    'scores': torch.zeros((0,), dtype=torch.float32, device=all_boxes.device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=all_boxes.device)
                })
        else:
            results.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=cls_preds[0].device),
                'scores': torch.zeros((0,), dtype=torch.float32, device=cls_preds[0].device),
                'labels': torch.zeros((0,), dtype=torch.long, device=cls_preds[0].device)
            })
    
    return results
