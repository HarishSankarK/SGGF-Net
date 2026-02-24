"""
Fusion-YOLOv11: Complete Multimodal RGB-Thermal Object Detection Model
Dual-stream SGGF-Net backbone + Mid-level fusion + PANet + YOLOv11 head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gfem import GFEM
from .fusion import MidLevelFusion
from .panet import PANet
from .yolo_head import YOLOv11Head
from .sggf_net import ResNetBackbone


class DualStreamSGGFNet(nn.Module):
    """
    Dual-Stream SGGF-Net Backbone
    Processes RGB and Thermal images in parallel streams
    """
    
    def __init__(self, pretrained=True, embed_dim=192, patch_size=32, 
                 num_layers=3, num_heads=6, backbone='resnet50'):
        """
        Args:
            pretrained: Whether to use pretrained ResNet weights
            embed_dim: GFEM embedding dimension
            patch_size: GFEM patch size
            num_layers: Number of transformer layers in GFEM
            num_heads: Number of attention heads in GFEM
            backbone: 'resnet50' or 'resnet101' (same C2–C5 channels, drop-in)
        """
        super(DualStreamSGGFNet, self).__init__()
        
        # RGB stream
        self.rgb_gfem = GFEM(
            in_channels=3, embed_dim=embed_dim, patch_size=patch_size,
            num_layers=num_layers, num_heads=num_heads
        )
        self.rgb_backbone = ResNetBackbone(pretrained=pretrained, backbone=backbone)
        self.rgb_fusion_conv = nn.Conv2d(embed_dim + 256, 256, kernel_size=1)
        
        # Thermal stream
        self.thermal_gfem = GFEM(
            in_channels=3, embed_dim=embed_dim, patch_size=patch_size,
            num_layers=num_layers, num_heads=num_heads
        )
        self.thermal_backbone = ResNetBackbone(pretrained=pretrained, backbone=backbone)
        self.thermal_fusion_conv = nn.Conv2d(embed_dim + 256, 256, kernel_size=1)
    
    def forward(self, rgb_images, thermal_images):
        """
        Args:
            rgb_images: RGB images [B, 3, H, W]
            thermal_images: Thermal images [B, 3, H, W]
        Returns:
            rgb_features: List of RGB feature maps [C2, C3, C4, C5]
            thermal_features: List of Thermal feature maps [C2, C3, C4, C5]
        """
        # RGB stream
        rgb_gfem_feat = self.rgb_gfem(rgb_images)
        rgb_resnet_feat = self.rgb_backbone(rgb_images)
        rgb_c2 = rgb_resnet_feat[0]
        
        # Fuse GFEM and ResNet for RGB
        rgb_gfem_upsampled = F.interpolate(
            rgb_gfem_feat, size=rgb_c2.shape[2:], 
            mode='bilinear', align_corners=False
        )
        rgb_fused_c2 = torch.cat([rgb_gfem_upsampled, rgb_c2], dim=1)
        rgb_fused_c2 = self.rgb_fusion_conv(rgb_fused_c2)
        
        rgb_features = [rgb_fused_c2] + rgb_resnet_feat[1:]
        
        # Thermal stream
        thermal_gfem_feat = self.thermal_gfem(thermal_images)
        thermal_resnet_feat = self.thermal_backbone(thermal_images)
        thermal_c2 = thermal_resnet_feat[0]
        
        # Fuse GFEM and ResNet for Thermal
        thermal_gfem_upsampled = F.interpolate(
            thermal_gfem_feat, size=thermal_c2.shape[2:],
            mode='bilinear', align_corners=False
        )
        thermal_fused_c2 = torch.cat([thermal_gfem_upsampled, thermal_c2], dim=1)
        thermal_fused_c2 = self.thermal_fusion_conv(thermal_fused_c2)
        
        thermal_features = [thermal_fused_c2] + thermal_resnet_feat[1:]
        
        return rgb_features, thermal_features


class FusionYOLOv11(nn.Module):
    """
    Complete Fusion-YOLOv11 Architecture
    
    Pipeline:
    1. Dual-stream SGGF-Net backbone (RGB + Thermal)
    2. Mid-level fusion (concatenation + cross-modal attention)
    3. PANet for multi-scale feature aggregation
    4. YOLOv11 detection head (one-stage, anchor-free)
    """
    
    def __init__(self, num_classes=80, pretrained=True, 
                 embed_dim=192, patch_size=32, num_layers=3, num_heads=6,
                 fusion_type='concat_attention', backbone='resnet50'):
        """
        Args:
            num_classes: Number of object classes
            pretrained: Whether to use pretrained ResNet weights
            embed_dim: GFEM embedding dimension
            patch_size: GFEM patch size
            num_layers: Number of transformer layers in GFEM
            num_heads: Number of attention heads in GFEM
            fusion_type: Fusion strategy ('concat', 'weighted', 'attention', 'concat_attention')
            backbone: 'resnet50' or 'resnet101' (ImageNet-pretrained, same feature channels)
        """
        super(FusionYOLOv11, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Dual-stream backbone
        self.dual_stream = DualStreamSGGFNet(
            pretrained=pretrained, embed_dim=embed_dim, patch_size=patch_size,
            num_layers=num_layers, num_heads=num_heads, backbone=backbone
        )
        
        # Mid-level fusion modules for each scale (different channel sizes)
        # C2: 256, C3: 512, C4: 1024, C5: 2048
        self.fusion_c2 = MidLevelFusion(in_channels=256, fusion_type=fusion_type)
        self.fusion_c3 = MidLevelFusion(in_channels=512, fusion_type=fusion_type)
        self.fusion_c4 = MidLevelFusion(in_channels=1024, fusion_type=fusion_type)
        self.fusion_c5 = MidLevelFusion(in_channels=2048, fusion_type=fusion_type)
        
        # PANet for multi-scale feature aggregation
        self.panet = PANet(in_channels_list=[256, 512, 1024, 2048], out_channels=256)
        
        # YOLOv11 detection head
        self.detection_head = YOLOv11Head(in_channels=256, num_classes=num_classes, num_anchors=1)
        
        # Loss function (created once, not per forward pass)
        from .yolo_head import YOLOv11Loss
        self.loss_fn = YOLOv11Loss(num_classes=num_classes)
    
    def forward(self, rgb_images, thermal_images, targets=None):
        """
        Args:
            rgb_images: RGB images [B, 3, H, W] or list of images
            thermal_images: Thermal images [B, 3, H, W] or list of images
            targets: Optional list of target dicts for training
        Returns:
            If training: loss_dict
            If inference: predictions dict
        """
        # Convert lists to tensors if needed
        if isinstance(rgb_images, list):
            rgb_images = torch.stack(rgb_images, dim=0)
        if isinstance(thermal_images, list):
            thermal_images = torch.stack(thermal_images, dim=0)
        
        # Dual-stream feature extraction
        rgb_features, thermal_features = self.dual_stream(rgb_images, thermal_images)
        
        # Mid-level fusion at each scale (using appropriate fusion module for each scale)
        fused_features = []
        fusion_modules = [self.fusion_c2, self.fusion_c3, self.fusion_c4, self.fusion_c5]
        for rgb_feat, thermal_feat, fusion_module in zip(rgb_features, thermal_features, fusion_modules):
            fused = fusion_module(rgb_feat, thermal_feat)
            fused_features.append(fused)
        
        # PANet for multi-scale aggregation
        panet_features = self.panet(fused_features)
        
        # YOLOv11 detection head
        predictions = self.detection_head(panet_features)
        
        if self.training and targets is not None:
            B, C, H, W = rgb_images.shape
            image_size = (H, W)
            loss_dict = self.loss_fn(predictions, targets, image_size=image_size)
            return loss_dict
        else:
            # Inference: return predictions
            return predictions
    
    def predict(self, rgb_images, thermal_images, conf_threshold=0.5, nms_threshold=0.5):
        """
        Inference method with post-processing
        
        Args:
            rgb_images: RGB images [B, 3, H, W] or list of images
            thermal_images: Thermal images [B, 3, H, W] or list of images
            conf_threshold: Confidence threshold for filtering
            nms_threshold: NMS IoU threshold
        Returns:
            List of detection results, each with 'boxes', 'scores', 'labels'
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(rgb_images, list):
                rgb_images = torch.stack(rgb_images, dim=0)
            if isinstance(thermal_images, list):
                thermal_images = torch.stack(thermal_images, dim=0)
            
            predictions = self.forward(rgb_images, thermal_images)
            
            B, C, H, W = rgb_images.shape
            image_size = (H, W)
            
            from .yolo_utils import generate_grid_points, post_process_predictions
            
            cls_preds = predictions['cls']
            num_scales = len(cls_preds)
            strides = [4 * (2 ** i) for i in range(num_scales)]
            grid_points_list = []
            
            device = cls_preds[0].device
            for scale_idx in range(num_scales):
                H_f, W_f = cls_preds[scale_idx].shape[2], cls_preds[scale_idx].shape[3]
                grid_points = generate_grid_points((H_f, W_f), strides[scale_idx], device=device)
                grid_points_list.append(grid_points)
            
            results = post_process_predictions(
                predictions, grid_points_list, strides,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                max_detections=100,
                image_size=image_size
            )
        
        if was_training:
            self.train()
        return results
