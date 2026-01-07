"""
Anchor generation and utilities for RPN
"""

import torch
import torch.nn as nn
import math


def generate_anchors(base_size=16, scales=[8, 16, 32], aspect_ratios=[0.5, 1.0, 2.0]):
    """
    Generate anchor boxes at a single location
    
    Args:
        base_size: Base size for anchors
        scales: List of scales
        aspect_ratios: List of aspect ratios
    Returns:
        anchors: (num_anchors, 4) tensor in [x_center, y_center, width, height] format
    """
    anchors = []
    for scale in scales:
        for aspect_ratio in aspect_ratios:
            w = base_size * scale * math.sqrt(aspect_ratio)
            h = base_size * scale / math.sqrt(aspect_ratio)
            # Anchor centered at (0, 0) - will be shifted to actual locations
            anchors.append([0, 0, w, h])
    
    return torch.tensor(anchors, dtype=torch.float32)


def generate_anchors_for_feature_map(feature_map_size, stride, base_size=16, 
                                     scales=[8, 16, 32], aspect_ratios=[0.5, 1.0, 2.0]):
    """
    Generate all anchor boxes for a feature map
    
    Args:
        feature_map_size: (H, W) of feature map
        stride: Stride of feature map relative to input image
        base_size: Base size for anchors
        scales: List of scales
        aspect_ratios: List of aspect ratios
    Returns:
        anchors: (H * W * num_anchors, 4) tensor in [x_center, y_center, width, height] format
    """
    H, W = feature_map_size
    num_anchors = len(scales) * len(aspect_ratios)
    
    # Generate base anchors
    base_anchors = generate_anchors(base_size, scales, aspect_ratios)  # (num_anchors, 4)
    
    # Generate grid of centers
    y_centers = torch.arange(0.5, H, 1.0) * stride
    x_centers = torch.arange(0.5, W, 1.0) * stride
    
    # Create meshgrid
    y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing='ij')
    centers = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)  # (H*W, 2)
    
    # Expand to all anchors
    centers = centers.unsqueeze(1).expand(H * W, num_anchors, 2)  # (H*W, num_anchors, 2)
    base_anchors = base_anchors.unsqueeze(0).expand(H * W, num_anchors, 4)  # (H*W, num_anchors, 4)
    
    # Shift anchors to grid locations
    anchors = base_anchors.clone()
    anchors[:, :, 0] = centers[:, :, 0]  # x_center
    anchors[:, :, 1] = centers[:, :, 1]  # y_center
    
    # Reshape to (H * W * num_anchors, 4)
    anchors = anchors.reshape(-1, 4)
    
    return anchors


def box_transform(anchors, deltas):
    """
    Apply bounding box deltas to anchors
    
    Args:
        anchors: (N, 4) in [x_center, y_center, width, height] format
        deltas: (N, 4) in [dx, dy, dw, dh] format
    Returns:
        boxes: (N, 4) transformed boxes in [x_center, y_center, width, height] format
    """
    # Extract anchor parameters
    anchor_x = anchors[:, 0]
    anchor_y = anchors[:, 1]
    anchor_w = anchors[:, 2]
    anchor_h = anchors[:, 3]
    
    # Extract deltas
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    
    # Apply transformation
    pred_x = anchor_x + dx * anchor_w
    pred_y = anchor_y + dy * anchor_h
    pred_w = anchor_w * torch.exp(dw)
    pred_h = anchor_h * torch.exp(dh)
    
    return torch.stack([pred_x, pred_y, pred_w, pred_h], dim=1)


def box_transform_inv(boxes, anchors):
    """
    Compute bounding box deltas from boxes to anchors
    
    Args:
        boxes: (N, 4) in [x_center, y_center, width, height] format
        anchors: (N, 4) in [x_center, y_center, width, height] format
    Returns:
        deltas: (N, 4) in [dx, dy, dw, dh] format
    """
    # Extract parameters
    box_x, box_y, box_w, box_h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    anchor_x, anchor_y, anchor_w, anchor_h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    
    # Compute deltas
    dx = (box_x - anchor_x) / (anchor_w + 1e-6)
    dy = (box_y - anchor_y) / (anchor_h + 1e-6)
    dw = torch.log(box_w / (anchor_w + 1e-6))
    dh = torch.log(box_h / (anchor_h + 1e-6))
    
    return torch.stack([dx, dy, dw, dh], dim=1)


def clip_boxes(boxes, image_size):
    """
    Clip boxes to image boundaries
    
    Args:
        boxes: (N, 4) in [x_center, y_center, width, height] format
        image_size: (H, W) of image
    Returns:
        clipped_boxes: (N, 4) clipped boxes
    """
    H, W = image_size
    clipped = boxes.clone()
    
    # Convert to corner format for clipping
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    
    # Clip
    x1 = torch.clamp(x1, 0, W)
    y1 = torch.clamp(y1, 0, H)
    x2 = torch.clamp(x2, 0, W)
    y2 = torch.clamp(y2, 0, H)
    
    # Convert back to center format
    clipped[:, 0] = (x1 + x2) / 2
    clipped[:, 1] = (y1 + y2) / 2
    clipped[:, 2] = x2 - x1
    clipped[:, 3] = y2 - y1
    
    return clipped


def nms(boxes, scores, iou_threshold=0.5, max_detections=100):
    """
    Non-Maximum Suppression
    
    Args:
        boxes: (N, 4) in [x_center, y_center, width, height] format
        scores: (N,) scores
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Convert to corner format
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    boxes_corners = torch.stack([x1, y1, x2, y2], dim=1)
    
    # Sort by score
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    boxes_sorted = boxes_corners[sorted_indices]
    
    keep = []
    while len(keep) < max_detections and len(boxes_sorted) > 0:
        # Keep the box with highest score
        keep.append(sorted_indices[0].item())
        
        if len(boxes_sorted) == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes_sorted[0:1]
        other_boxes = boxes_sorted[1:]
        
        # Calculate IoU
        inter_x1 = torch.max(current_box[:, 0], other_boxes[:, 0])
        inter_y1 = torch.max(current_box[:, 1], other_boxes[:, 1])
        inter_x2 = torch.min(current_box[:, 2], other_boxes[:, 2])
        inter_y2 = torch.min(current_box[:, 3], other_boxes[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        area_current = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        area_other = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union_area = area_current + area_other - inter_area
        
        ious = inter_area / (union_area + 1e-6)
        
        # Keep boxes with IoU < threshold
        mask = ious < iou_threshold
        boxes_sorted = boxes_sorted[1:][mask]
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

