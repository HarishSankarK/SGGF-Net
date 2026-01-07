"""
SGGF-Net: Main network architecture
UAV Image Object Detection based on Self-Attention Guidance and Global Feature Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from .gfem import GFEM
from .ndpa import NDPA
from .arpm import ARPM, ROIAlign


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from different levels
        Returns:
            List of FPN feature maps
        """
        # Top-down pathway
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down features
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='nearest'
            )
        
        # Apply output convolutions
        outputs = [output_conv(lateral) for output_conv, lateral in zip(self.output_convs, laterals)]
        
        # Add extra level (P6) by max pooling P5
        if len(outputs) > 0:
            p6 = F.max_pool2d(outputs[-1], kernel_size=1, stride=2)
            outputs.append(p6)
        
        return outputs


class RPNHead(nn.Module):
    """Region Proposal Network Head"""
    
    def __init__(self, in_channels=256, num_anchors=3):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from FPN
        Returns:
            cls_logits: Classification logits for each anchor
            bbox_pred: Bounding box predictions
        """
        cls_logits_list = []
        bbox_pred_list = []
        
        for feat in features:
            x = F.relu(self.conv(feat))
            cls_logits_list.append(self.cls_logits(x))
            bbox_pred_list.append(self.bbox_pred(x))
        
        return cls_logits_list, bbox_pred_list


class ROIHead(nn.Module):
    """ROI Head for classification and bounding box regression"""
    
    def __init__(self, in_channels=256, num_classes=11, representation_size=1024):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        
        # Two FC layers
        self.fc6 = nn.Linear(in_channels * 7 * 7, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        
        # Classification and regression heads
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)
        
    def forward(self, roi_features):
        """
        Args:
            roi_features: ROI features (N, C, 7, 7)
        Returns:
            cls_scores: Classification scores (N, num_classes)
            bbox_pred: Bounding box predictions (N, num_classes * 4)
        """
        # Flatten
        x = roi_features.view(roi_features.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        # Outputs
        cls_scores = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_scores, bbox_pred


class SGGFNet(nn.Module):
    """
    SGGF-Net: UAV Image Object Detection Network
    
    Architecture:
    1. Backbone with GFEM (Global Feature Extraction Module)
    2. FPN (Feature Pyramid Network)
    3. RPN Head with NDPA (Normal Distribution-based Prior Assigner)
    4. ROI Extractor with ARPM (Attention-guided ROI Pooling Module)
    5. ROI Head for final classification and regression
    """
    
    def __init__(self, num_classes=11, pretrained=True):
        """
        Args:
            num_classes: Number of object classes (10 classes + background = 11)
            pretrained: Whether to use pretrained ResNet weights
        """
        super(SGGFNet, self).__init__()
        self.num_classes = num_classes
        
        # GFEM for global feature extraction
        self.gfem = GFEM(in_channels=3, embed_dim=256, patch_size=4, num_layers=4)
        
        # ResNet backbone (we'll use ResNet50 as base)
        # Note: In practice, GFEM output would be integrated with ResNet features
        # For simplicity, we use a hybrid approach
        if pretrained:
            backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        else:
            backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        
        # FPN for multi-scale features
        # Assuming ResNet outputs features at 4 scales
        self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        
        # RPN Head with NDPA
        self.rpn_head = RPNHead(in_channels=256, num_anchors=3)
        self.ndpa = NDPA(pos_threshold=0.5, neg_threshold=0.3)
        
        # ROI Extractor with ARPM
        self.roi_align = ROIAlign(output_size=(7, 7), spatial_scale=1.0/16.0)
        self.arpm = ARPM(in_channels=256, out_channels=256, roi_size=7)
        
        # ROI Head
        self.roi_head = ROIHead(in_channels=256, num_classes=num_classes)
        
    def forward(self, images, targets=None):
        """
        Args:
            images: List of images or batched tensor (B, C, H, W)
            targets: List of target dicts with 'boxes' and 'labels' keys
        Returns:
            If training: loss dict
            If inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")
        
        # Extract features using GFEM
        gfem_features = self.gfem(images)
        
        # For now, we'll use a simplified approach where GFEM features
        # are combined with ResNet features. In full implementation,
        # this would be more sophisticated.
        
        # Get ResNet features (simplified - in practice, integrate with GFEM)
        # For this implementation, we'll use a basic feature extraction
        # In production, you'd integrate GFEM output with ResNet backbone
        
        # FPN features
        # Note: This is simplified. Full implementation would extract
        # features from ResNet backbone and combine with GFEM
        
        if self.training:
            return self._forward_train(images, targets, gfem_features)
        else:
            return self._forward_inference(images, gfem_features)
    
    def _forward_train(self, images, targets, gfem_features):
        """Forward pass during training"""
        # This is a simplified training forward pass
        # Full implementation would include:
        # 1. Feature extraction from backbone + GFEM
        # 2. RPN proposals with NDPA
        # 3. ROI extraction with ARPM
        # 4. Loss computation
        
        losses = {}
        # Placeholder for actual loss computation
        losses['loss_classifier'] = torch.tensor(0.0, device=images.device)
        losses['loss_box_reg'] = torch.tensor(0.0, device=images.device)
        losses['loss_objectness'] = torch.tensor(0.0, device=images.device)
        losses['loss_rpn_box_reg'] = torch.tensor(0.0, device=images.device)
        
        return losses
    
    def _forward_inference(self, images, gfem_features):
        """Forward pass during inference"""
        # Placeholder for inference
        # Full implementation would return detections
        batch_size = images.size(0) if isinstance(images, torch.Tensor) else len(images)
        results = []
        
        for i in range(batch_size):
            results.append({
                'boxes': torch.zeros(0, 4, device=images.device),
                'labels': torch.zeros(0, dtype=torch.long, device=images.device),
                'scores': torch.zeros(0, device=images.device)
            })
        
        return results

