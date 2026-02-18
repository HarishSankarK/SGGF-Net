"""
YOLOv11 Detection Head
One-stage anchor-free object detection head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .yolo_utils import assign_targets_to_predictions, generate_grid_points, decode_bbox


class YOLOv11Head(nn.Module):
    """
    YOLOv11 Detection Head
    One-stage anchor-free detector that directly predicts bounding boxes and classes
    """
    
    def __init__(self, in_channels=256, num_classes=80, num_anchors=1):
        """
        Args:
            in_channels: Input feature channels (from FPN/PANet)
            num_classes: Number of object classes
            num_anchors: Number of anchors per location (YOLOv11 uses 1, anchor-free)
        """
        super(YOLOv11Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolution layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes * num_anchors, kernel_size=1)
        )
        
        # Objectness head (confidence score)
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        )
        
        # Bounding box regression head (x, y, w, h)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1)  # 4 for (x, y, w, h)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper priors for fast convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize objectness head final layer bias to predict "no object" initially
        # sigmoid(-4.6) ≈ 0.01 — model starts by saying "no object" everywhere
        # and only learns to activate for real objects
        obj_final_conv = self.obj_head[-1]
        nn.init.constant_(obj_final_conv.bias, -4.6)
        
        # Initialize classification head final layer bias similarly
        cls_final_conv = self.cls_head[-1]
        nn.init.constant_(cls_final_conv.bias, -4.6)
        
        # Initialize bbox dw/dh bias so initial predicted sizes are reasonable.
        # Output channels [dx, dy, dw, dh]: exp(2.0)*stride gives ~7.4*stride,
        # so P3(8)→59px, P4(16)→118px — close to typical person/vehicle sizes.
        bbox_final_conv = self.bbox_head[-1]
        with torch.no_grad():
            bbox_final_conv.bias[2] = 2.0  # dw
            bbox_final_conv.bias[3] = 2.0  # dh
    
    def forward(self, features):
        """
        Args:
            features: List of multi-scale feature maps from FPN/PANet
                    Each element: [B, C, H, W]
        Returns:
            predictions: Dict with keys:
                - 'cls': List of classification logits [B, num_classes, H, W] for each scale
                - 'obj': List of objectness scores [B, 1, H, W] for each scale
                - 'bbox': List of bbox predictions [B, 4, H, W] for each scale
        """
        cls_preds = []
        obj_preds = []
        bbox_preds = []
        
        for feat in features:
            # Shared feature extraction
            shared_feat = self.shared_conv(feat)
            
            # Classification prediction
            cls_pred = self.cls_head(shared_feat)
            B, C, H, W = cls_pred.shape
            cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W)
            cls_pred = cls_pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_anchors, H, W, num_classes]
            cls_preds.append(cls_pred)
            
            # Objectness prediction
            obj_pred = self.obj_head(shared_feat)
            obj_pred = obj_pred.view(B, self.num_anchors, 1, H, W)
            obj_pred = obj_pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_anchors, H, W, 1]
            obj_preds.append(obj_pred)
            
            # Bounding box prediction
            bbox_pred = self.bbox_head(shared_feat)
            bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W)
            bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_anchors, H, W, 4]
            bbox_preds.append(bbox_pred)
        
        return {
            'cls': cls_preds,
            'obj': obj_preds,
            'bbox': bbox_preds
        }


class YOLOv11Loss(nn.Module):
    """
    YOLOv11 Loss Function
    Combines classification, objectness, and bounding box regression losses
    """
    
    def __init__(self, num_classes=80, obj_weight=1.0, cls_weight=1.0, bbox_weight=5.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        """
        Args:
            num_classes: Number of classes
            obj_weight: Weight for objectness loss
            cls_weight: Weight for classification loss
            bbox_weight: Weight for bounding box regression loss
            focal_alpha: Focal loss alpha (balances pos/neg)
            focal_gamma: Focal loss gamma (focuses on hard examples)
        """
        super(YOLOv11Loss, self).__init__()
        self.num_classes = num_classes
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def focal_loss(self, logits, targets):
        """Focal loss: down-weights easy examples, focuses on hard ones"""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        return (focal_weight * bce).mean()
        
    def ciou_loss(self, pred_boxes, target_boxes):
        """
        Complete IoU (CIoU) Loss for bounding box regression
        """
        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
        
        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
        
        # Calculate IoU
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-7)
        
        # CIoU components
        # Center distance
        pred_center_x = pred_boxes[..., 0]
        pred_center_y = pred_boxes[..., 1]
        target_center_x = target_boxes[..., 0]
        target_center_y = target_boxes[..., 1]
        
        center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Enclosing box diagonal
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # Aspect ratio consistency
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_boxes[..., 2] / (target_boxes[..., 3] + 1e-7)) -
            torch.atan(pred_boxes[..., 2] / (pred_boxes[..., 3] + 1e-7)), 2
        )
        alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU
        ciou = iou - (center_distance / (enclose_diagonal + 1e-7)) - alpha * v
        
        return 1 - ciou
    
    def forward(self, predictions, targets, image_size=(640, 640)):
        """
        Args:
            predictions: Dict with 'cls', 'obj', 'bbox' predictions
            targets: List of target dicts, each with 'boxes' and 'labels'
            image_size: (H, W) of input image
        Returns:
            loss_dict: Dict with individual losses
        """
        cls_preds = predictions['cls']  # List of (B, num_anchors, H, W, num_classes)
        obj_preds = predictions['obj']  # List of (B, num_anchors, H, W, 1)
        bbox_preds = predictions['bbox']  # List of (B, num_anchors, H, W, 4)
        
        B = cls_preds[0].shape[0]
        num_scales = len(cls_preds)
        device = cls_preds[0].device
        
        # Generate grid points and strides for each scale
        # Strides: [4, 8, 16, 32, 64] for P2, P3, P4, P5, P6
        # ResNet: C2=1/4, C3=1/8, C4=1/16, C5=1/32; P6=max_pool(P5)=1/64
        strides = [4 * (2 ** i) for i in range(num_scales)]
        grid_points_list = []
        
        for scale_idx in range(num_scales):
            H, W = cls_preds[scale_idx].shape[2], cls_preds[scale_idx].shape[3]
            grid_points = generate_grid_points((H, W), strides[scale_idx], device=device)
            grid_points_list.append(grid_points)
        
        # Assign targets to predictions
        # Squeeze num_anchors dimension (anchor-free: num_anchors=1)
        # bbox_preds: List of (B, num_anchors, H, W, 4) -> (B, H, W, 4)
        bbox_preds_squeezed = [pred.squeeze(1) for pred in bbox_preds]
        
        assigned_targets = assign_targets_to_predictions(
            bbox_preds_squeezed, grid_points_list, strides, targets, image_size,
            pos_threshold=0.5, neg_threshold=0.4
        )
        
        # Compute losses (tensors on correct device for consistent return types)
        zero = torch.zeros(1, device=device).squeeze()
        total_obj_loss = zero.clone()
        total_cls_loss = zero.clone()
        total_bbox_loss = zero.clone()
        num_positives = 0
        
        for scale_idx in range(num_scales):
            cls_pred = cls_preds[scale_idx]  # (B, num_anchors, H, W, num_classes)
            obj_pred = obj_preds[scale_idx]  # (B, num_anchors, H, W, 1)
            bbox_pred = bbox_preds[scale_idx]  # (B, num_anchors, H, W, 4)
            
            # Take first anchor (anchor-free)
            cls_pred = cls_pred[:, 0]  # (B, H, W, num_classes)
            obj_pred = obj_pred[:, 0, ..., 0]  # (B, H, W)
            bbox_pred = bbox_pred[:, 0]  # (B, H, W, 4)
            
            assigned = assigned_targets[scale_idx]
            obj_targets = assigned['obj_targets']  # (B, H, W)
            cls_targets = assigned['cls_targets']  # (B, H, W)
            bbox_targets = assigned['bbox_targets']  # (B, H, W, 4)
            
            # Objectness loss (focal loss — focuses on hard examples)
            obj_loss = self.focal_loss(obj_pred, obj_targets)
            total_obj_loss += obj_loss
            
            # Classification loss (focal loss per class, positives only)
            pos_mask = obj_targets > 0.5  # (B, H, W)
            if pos_mask.any():
                cls_pred_pos = cls_pred[pos_mask]  # (N_pos, num_classes)
                cls_targets_pos = cls_targets[pos_mask]  # (N_pos,)
                
                cls_targets_onehot = torch.zeros_like(cls_pred_pos)
                cls_targets_onehot.scatter_(1, cls_targets_pos.unsqueeze(1), 1.0)
                
                cls_loss = self.focal_loss(cls_pred_pos, cls_targets_onehot)
                total_cls_loss += cls_loss
                num_positives += pos_mask.sum().item()
            
            # Bounding box loss (CIoU, only for positives)
            if pos_mask.any():
                bbox_pred_pos = bbox_pred[pos_mask]  # (N_pos, 4) raw predictions
                bbox_targets_pos = bbox_targets[pos_mask]  # (N_pos, 4) dx,dy,dw,dh targets
                
                grid_points = grid_points_list[scale_idx]
                stride = strides[scale_idx]
                
                # Get grid points for positive locations
                pos_indices_2d = torch.where(pos_mask)
                if len(pos_indices_2d) == 3:
                    pos_b, pos_y, pos_x = pos_indices_2d[0], pos_indices_2d[1], pos_indices_2d[2]
                    pos_grid_points = grid_points[pos_y, pos_x]  # (N_pos, 2)
                else:
                    pos_grid_points = grid_points.reshape(-1, 2)[pos_mask.reshape(-1)]
                
                # Decode predictions to absolute center format
                decoded_pred = decode_bbox(bbox_pred_pos, pos_grid_points, stride)  # (N_pos, 4) cx,cy,w,h
                
                # Decode targets: targets are deltas relative to grid cell
                # dx, dy are relative to stride; dw, dh are log-scale relative to stride
                target_cx = pos_grid_points[:, 0] + bbox_targets_pos[:, 0] * stride
                target_cy = pos_grid_points[:, 1] + bbox_targets_pos[:, 1] * stride
                target_w = torch.exp(bbox_targets_pos[:, 2]) * stride
                target_h = torch.exp(bbox_targets_pos[:, 3]) * stride
                decoded_target = torch.stack([target_cx, target_cy, target_w, target_h], dim=1)
                
                bbox_loss = self.ciou_loss(decoded_pred, decoded_target)
                bbox_loss = bbox_loss.mean()
                total_bbox_loss += bbox_loss
        
        # Average across scales
        total_obj_loss = total_obj_loss / num_scales
        total_cls_loss = total_cls_loss / num_scales
        total_bbox_loss = total_bbox_loss / num_scales
        
        # Weighted combination
        total_loss = (self.obj_weight * total_obj_loss + 
                     self.cls_weight * total_cls_loss + 
                     self.bbox_weight * total_bbox_loss)
        
        return {
            'total_loss': total_loss,
            'obj_loss': total_obj_loss,
            'cls_loss': total_cls_loss,
            'bbox_loss': total_bbox_loss
        }
