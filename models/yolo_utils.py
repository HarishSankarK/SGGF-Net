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
    # Handle both full feature map and individual predictions
    if predictions.dim() == 4:
        # Full feature map: (B, H, W, 4)
        B, H, W, _ = predictions.shape
        # grid_points may be (H, W, 2) or (1, H, W, 2) - squash to (H, W, 2)
        if grid_points.dim() == 4:
            grid_points = grid_points.squeeze(0)
        grid = grid_points.unsqueeze(0).expand(B, H, W, 2)  # (B, H, W, 2)
        
        # Decode predictions
        dx, dy, dw, dh = predictions[..., 0], predictions[..., 1], predictions[..., 2], predictions[..., 3]
        
        # Apply sigmoid to dx, dy (relative to grid cell)
        dx = torch.sigmoid(dx) * stride
        dy = torch.sigmoid(dy) * stride
        
        # Apply exp to dw, dh (scale factors)
        dw = torch.exp(torch.clamp(dw, max=10)) * stride
        dh = torch.exp(torch.clamp(dh, max=10)) * stride
        
        # Decode center coordinates
        x_center = grid[..., 0] + dx
        y_center = grid[..., 1] + dy
        
        boxes = torch.stack([x_center, y_center, dw, dh], dim=-1)
        
        return boxes
    else:
        # Individual predictions: (N, 4)
        N = predictions.shape[0]
        # grid_points should be (N, 2)
        if grid_points.dim() == 2 and grid_points.shape[0] == N:
            grid = grid_points  # (N, 2)
        else:
            raise ValueError(f"grid_points shape {grid_points.shape} doesn't match predictions shape {predictions.shape}")
        
        # Decode predictions
        dx, dy, dw, dh = predictions[..., 0], predictions[..., 1], predictions[..., 2], predictions[..., 3]
        
        # Apply sigmoid to dx, dy (relative to grid cell)
        dx = torch.sigmoid(dx) * stride
        dy = torch.sigmoid(dy) * stride
        
        # Apply exp to dw, dh (scale factors)
        dw = torch.exp(torch.clamp(dw, max=10)) * stride
        dh = torch.exp(torch.clamp(dh, max=10)) * stride
        
        # Decode center coordinates
        x_center = grid[..., 0] + dx
        y_center = grid[..., 1] + dy
        
        boxes = torch.stack([x_center, y_center, dw, dh], dim=-1)
        
        return boxes


def compute_iou(boxes1, boxes2, chunk_size=8192):
    """
    Compute IoU between two sets of boxes. Uses chunking to avoid OOM on large matrices.
    
    Args:
        boxes1: (N, 4) in (x_center, y_center, w, h) format
        boxes2: (M, 4) in (x_center, y_center, w, h) format
        chunk_size: Process boxes1 in chunks to limit memory (default 8192)
    Returns:
        iou: (N, M) IoU matrix
    """
    def to_corners(boxes):
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    N, M = boxes1.shape[0], boxes2.shape[0]
    # Chunk boxes1 to avoid (N, M) matrix OOM when N*M is large (e.g. 36k*100=3.6M floats)
    if N <= chunk_size:
        # Small enough - compute directly
        boxes1_corners = to_corners(boxes1)
        boxes2_corners = to_corners(boxes2)
        boxes1_exp = boxes1_corners.unsqueeze(1)
        boxes2_exp = boxes2_corners.unsqueeze(0)
        inter_x1 = torch.max(boxes1_exp[..., 0], boxes2_exp[..., 0])
        inter_y1 = torch.max(boxes1_exp[..., 1], boxes2_exp[..., 1])
        inter_x2 = torch.min(boxes1_exp[..., 2], boxes2_exp[..., 2])
        inter_y2 = torch.min(boxes1_exp[..., 3], boxes2_exp[..., 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        area1 = torch.clamp(boxes1_corners[:, 2] - boxes1_corners[:, 0], min=0) * \
                torch.clamp(boxes1_corners[:, 3] - boxes1_corners[:, 1], min=0)
        area2 = torch.clamp(boxes2_corners[:, 2] - boxes2_corners[:, 0], min=0) * \
                torch.clamp(boxes2_corners[:, 3] - boxes2_corners[:, 1], min=0)
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        union_area = torch.clamp(union_area, min=1e-7)
        return inter_area / union_area
    
    iou_list = []
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = compute_iou(boxes1[i:end], boxes2, chunk_size=chunk_size * 2)
        iou_list.append(chunk)
    return torch.cat(iou_list, dim=0)


def assign_targets_to_predictions(predictions_list, grid_points_list, strides, targets, image_size, 
                                  pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign ground truth targets to anchor-free predictions
    
    Args:
        predictions_list: List of bbox predictions for each scale [(B, H_i, W_i, 4), ...]
        grid_points_list: List of grid points for each scale [(H_i, W_i, 2), ...]
        strides: List of strides for each scale
        targets: List of target dicts with 'boxes' and 'labels'
        image_size: (H, W) of input image
        pos_threshold: IoU threshold for positive samples
        neg_threshold: IoU threshold for negative samples
    Returns:
        assigned_targets: List of dicts with 'labels', 'bbox_targets', 'obj_targets' for each scale
    """
    B = len(targets)
    num_scales = len(predictions_list)
    
    assigned_targets = []
    
    for scale_idx in range(num_scales):
        predictions = predictions_list[scale_idx]  # (B, H, W, 4)
        grid_points = grid_points_list[scale_idx]  # (H, W, 2)
        stride = strides[scale_idx]
        
        B, H, W, _ = predictions.shape
        
        # Decode predictions to get current boxes
        decoded_boxes = decode_bbox(predictions, grid_points, stride)  # (B, H, W, 4)
        decoded_boxes_flat = decoded_boxes.reshape(B, H * W, 4)  # (B, H*W, 4)
        
        # Initialize targets
        labels = torch.zeros(B, H, W, dtype=torch.long, device=predictions.device)
        bbox_targets = torch.zeros(B, H, W, 4, dtype=torch.float32, device=predictions.device)
        obj_targets = torch.zeros(B, H, W, dtype=torch.float32, device=predictions.device)
        cls_targets = torch.zeros(B, H, W, dtype=torch.long, device=predictions.device)
        
        max_gt_boxes = 100  # Limit to avoid OOM on dense images
        for b in range(B):
            target = targets[b]
            gt_boxes = target['boxes']  # (M, 4) in [x1, y1, x2, y2]
            gt_labels = target['labels']  # (M,)

            # Sanitize GT boxes: enforce proper corner ordering and finite values.
            if len(gt_boxes) > 0:
                x1 = torch.min(gt_boxes[:, 0], gt_boxes[:, 2])
                y1 = torch.min(gt_boxes[:, 1], gt_boxes[:, 3])
                x2 = torch.max(gt_boxes[:, 0], gt_boxes[:, 2])
                y2 = torch.max(gt_boxes[:, 1], gt_boxes[:, 3])
                gt_boxes = torch.stack([x1, y1, x2, y2], dim=1)

                if image_size is not None and len(image_size) == 2:
                    img_h, img_w = image_size
                    gt_boxes[:, 0] = gt_boxes[:, 0].clamp(0, img_w)
                    gt_boxes[:, 2] = gt_boxes[:, 2].clamp(0, img_w)
                    gt_boxes[:, 1] = gt_boxes[:, 1].clamp(0, img_h)
                    gt_boxes[:, 3] = gt_boxes[:, 3].clamp(0, img_h)

                gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                valid_mask = torch.isfinite(gt_boxes).all(dim=1) & (gt_w > 1e-4) & (gt_h > 1e-4)
                gt_boxes = gt_boxes[valid_mask]
                gt_labels = gt_labels[valid_mask]
            if len(gt_boxes) > max_gt_boxes:
                perm = torch.randperm(len(gt_boxes), device=gt_boxes.device)[:max_gt_boxes]
                gt_boxes = gt_boxes[perm]
                gt_labels = gt_labels[perm]
            if len(gt_boxes) == 0:
                # No ground truth - all negatives
                continue
            
            # Convert GT boxes to center format
            x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
            gt_cx = (x1 + x2) / 2
            gt_cy = (y1 + y2) / 2
            gt_w = x2 - x1
            gt_h = y2 - y1
            gt_boxes_center = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)  # (M, 4)
            
            # Compute IoU between predictions and GT
            pred_boxes_flat = decoded_boxes_flat[b]  # (H*W, 4)
            iou_matrix = compute_iou(pred_boxes_flat, gt_boxes_center)  # (H*W, M)
            iou_matrix = torch.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Find best GT match for each prediction
            max_iou, matched_gt_idx = iou_matrix.max(dim=1)  # (H*W,)

            # Base assignment from IoU thresholds
            pos_mask = max_iou >= pos_threshold
            neg_mask = max_iou < neg_threshold

            # Stability fix: force at least one positive prediction per GT.
            best_pred_per_gt = iou_matrix.argmax(dim=0)  # (M,)
            force_pos_mask = torch.zeros_like(pos_mask)
            force_pos_mask[best_pred_per_gt] = True
            pos_mask = pos_mask | force_pos_mask
            neg_mask = neg_mask & (~pos_mask)

            # Ensure forced positives are matched to corresponding GT indices.
            assigned_gt_idx = matched_gt_idx.clone()
            assigned_gt_idx[best_pred_per_gt] = torch.arange(
                gt_boxes_center.shape[0], device=gt_boxes_center.device
            )

            labels[b][pos_mask.reshape(H, W)] = gt_labels[assigned_gt_idx[pos_mask]]
            obj_targets[b][pos_mask.reshape(H, W)] = 1.0
            obj_targets[b][neg_mask.reshape(H, W)] = 0.0
            
            # Assign bbox targets (only for positives)
            if pos_mask.any():
                matched_gt = gt_boxes_center[assigned_gt_idx[pos_mask]]  # (N_pos, 4)
                matched_pred = pred_boxes_flat[pos_mask]  # (N_pos, 4)
                
                # Compute bbox deltas
                dx = (matched_gt[:, 0] - matched_pred[:, 0]) / (matched_pred[:, 2] + 1e-6)
                dy = (matched_gt[:, 1] - matched_pred[:, 1]) / (matched_pred[:, 3] + 1e-6)
                dw = torch.log((matched_gt[:, 2] + 1e-6) / (matched_pred[:, 2] + 1e-6))
                dh = torch.log((matched_gt[:, 3] + 1e-6) / (matched_pred[:, 3] + 1e-6))
                
                bbox_deltas = torch.stack([dx, dy, dw, dh], dim=1)
                
                # Reshape and assign
                pos_indices = torch.where(pos_mask.reshape(H, W))
                bbox_targets[b][pos_indices] = bbox_deltas
                cls_targets[b][pos_indices] = gt_labels[assigned_gt_idx[pos_mask]]
        
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
        results: List of detection dicts, each with 'boxes', 'scores', 'labels'
    """
    cls_preds = predictions['cls']  # List of (B, num_anchors, H, W, num_classes)
    obj_preds = predictions['obj']  # List of (B, num_anchors, H, W, 1)
    bbox_preds = predictions['bbox']  # List of (B, num_anchors, H, W, 4)
    
    B = cls_preds[0].shape[0]
    num_scales = len(cls_preds)
    
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
            
            num_anchors, H, W, num_classes = cls_pred.shape
            
            # For anchor-free, num_anchors is typically 1
            # Take first anchor
            cls_pred = cls_pred[0]  # (H, W, num_classes)
            obj_pred = obj_pred[0, ..., 0]  # (H, W)
            bbox_pred = bbox_pred[0]  # (H, W, 4)
            
            # Apply sigmoid to get probabilities
            cls_prob = torch.sigmoid(cls_pred)  # (H, W, num_classes)
            obj_prob = torch.sigmoid(obj_pred)  # (H, W)
            
            # Combined confidence
            conf = obj_prob.unsqueeze(-1) * cls_prob  # (H, W, num_classes)
            
            # Decode boxes (grid_points is (H, W, 2); decode_bbox expects that and adds batch dim internally)
            boxes = decode_bbox(bbox_pred.unsqueeze(0), grid_points, stride)[0]  # (H, W, 4)
            
            # Filter by confidence
            mask = conf > conf_threshold  # (H, W, num_classes)
            
            if mask.any():
                # Get indices of confident predictions
                y_indices, x_indices, label_indices = torch.where(mask)
                
                # Get boxes, scores, labels
                boxes_selected = boxes[y_indices, x_indices]  # (N, 4)
                scores_selected = conf[y_indices, x_indices, label_indices]  # (N,)
                labels_selected = label_indices  # (N,)
                
                all_boxes.append(boxes_selected)
                all_scores.append(scores_selected)
                all_labels.append(labels_selected)
        
        if len(all_boxes) > 0:
            # Concatenate all scales
            all_boxes = torch.cat(all_boxes, dim=0)  # (N_total, 4)
            all_scores = torch.cat(all_scores, dim=0)  # (N_total,)
            all_labels = torch.cat(all_labels, dim=0)  # (N_total,)
            
            # Apply NMS per class
            from .anchor_utils import nms, center_to_corners
            
            # Convert boxes to center format for NMS
            all_boxes_center = all_boxes  # Already in center format
            
            final_boxes = []
            final_scores = []
            final_labels = []
            
            # Skip background class (0) in final detections.
            for cls_id in range(1, num_classes):
                cls_mask = all_labels == cls_id
                if not cls_mask.any():
                    continue
                
                cls_boxes = all_boxes[cls_mask]
                cls_scores = all_scores[cls_mask]
                
                if len(cls_boxes) == 0:
                    continue
                
                # Apply NMS (nms expects center format)
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
                
                # Convert boxes to corner format for output
                x_center, y_center, w, h = final_boxes[:, 0], final_boxes[:, 1], final_boxes[:, 2], final_boxes[:, 3]
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                boxes_corners = torch.stack([x1, y1, x2, y2], dim=1)
                
                results.append({
                    'boxes': boxes_corners,
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
            # No detections
            results.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=cls_preds[0].device),
                'scores': torch.zeros((0,), dtype=torch.float32, device=cls_preds[0].device),
                'labels': torch.zeros((0,), dtype=torch.long, device=cls_preds[0].device)
            })
    
    return results
