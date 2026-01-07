"""
Attention-guided ROI Pooling Module (ARPM)

Optimizes multi-scale feature fusion using attention mechanism for ROI pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARPM(nn.Module):
    """
    Attention-guided ROI Pooling Module
    
    Integrates sampling results from parallel feature fusion paths using
    self-attention mechanism to optimize feature map quality.
    """
    
    def __init__(self, in_channels=256, out_channels=256, roi_size=7):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            roi_size: Size of ROI output (typically 7x7)
        """
        super(ARPM, self).__init__()
        self.roi_size = roi_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Stage 1: Deep fusion using multiple conv kernels
        # Four 5x5 conv kernels for feature fusion
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1)
            for _ in range(4)
        ])
        
        # Stage 2: Query, Key, Value projections for self-attention
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Stage 3: Output projection
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, roi_features):
        """
        Args:
            roi_features: ROI features from ROI Align, shape (N, C, H, W)
                         where N is number of ROIs, typically (N, 256, 7, 7)
        Returns:
            Enhanced features of shape (N, out_channels, H, W)
        """
        # Stage 1: Deep fusion using multiple conv kernels
        # Apply four 5x5 convolutions and sum them
        fused_features = None
        for conv in self.fusion_convs:
            conv_out = conv(roi_features)
            if fused_features is None:
                fused_features = conv_out
            else:
                fused_features = fused_features + conv_out
        
        # Stage 2: Self-attention mechanism
        # Generate Q, K, V
        query = self.query_conv(fused_features)  # (N, C, H, W)
        key = self.key_conv(fused_features)      # (N, C, H, W)
        value = self.value_conv(fused_features) # (N, C, H, W)
        
        # Reshape for attention computation: (N, C, H, W) -> (N, C, H*W)
        N, C, H, W = query.shape
        query = query.view(N, C, H * W)  # (N, C, H*W)
        key = key.view(N, C, H * W)      # (N, C, H*W)
        value = value.view(N, C, H * W)   # (N, C, H*W)
        
        # Compute attention: Q * K^T / sqrt(C)
        attention_scores = torch.bmm(query.transpose(1, 2), key) / (C ** 0.5)  # (N, H*W, H*W)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (N, H*W, H*W)
        
        # Apply attention to values
        attended_value = torch.bmm(value, attention_weights.transpose(1, 2))  # (N, C, H*W)
        
        # Reshape back: (N, C, H*W) -> (N, C, H, W)
        attended_value = attended_value.view(N, C, H, W)
        
        # Stage 3: Generate output features
        # As per paper: features = Conv2D(ReLU(value) * (key * Softmax(query)), 1, 1)
        # Simplified version: apply attention-weighted value through output conv
        enhanced_features = F.relu(attended_value) * value.view(N, C, H, W)
        output = self.output_conv(enhanced_features)
        
        return output


class ROIAlign(nn.Module):
    """
    ROI Align layer for extracting fixed-size features from ROIs
    """
    
    def __init__(self, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=-1):
        """
        Args:
            output_size: Output size (height, width)
            spatial_scale: Scale factor for ROI coordinates
            sampling_ratio: Number of sampling points (-1 for adaptive)
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        
    def forward(self, features, rois):
        """
        Args:
            features: Feature maps (B, C, H, W)
            rois: ROI boxes (N, 5) where first column is batch index, rest are [x1, y1, x2, y2]
        Returns:
            ROI features (N, C, output_size[0], output_size[1])
        """
        # Use torchvision's ROIAlign if available, otherwise use bilinear interpolation
        try:
            from torchvision.ops import roi_align
            return roi_align(
                features, rois, 
                output_size=self.output_size,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio
            )
        except ImportError:
            # Fallback: simple bilinear interpolation
            return self._simple_roi_align(features, rois)
    
    def _simple_roi_align(self, features, rois):
        """Simple ROI align using bilinear interpolation"""
        N = rois.size(0)
        C = features.size(1)
        H_out, W_out = self.output_size
        
        output = torch.zeros(N, C, H_out, W_out, device=features.device, dtype=features.dtype)
        
        for i in range(N):
            batch_idx = int(rois[i, 0])
            x1, y1, x2, y2 = rois[i, 1:].tolist()
            
            # Scale coordinates
            x1 = x1 * self.spatial_scale
            y1 = y1 * self.spatial_scale
            x2 = x2 * self.spatial_scale
            y2 = y2 * self.spatial_scale
            
            # Extract ROI region
            roi_h = max(1, y2 - y1)
            roi_w = max(1, x2 - x1)
            
            # Sample points in output grid
            roi_feature = features[batch_idx, :, int(y1):int(y2)+1, int(x1):int(x2)+1]
            if roi_feature.numel() > 0:
                roi_feature = F.interpolate(
                    roi_feature.unsqueeze(0),
                    size=(H_out, W_out),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                output[i] = roi_feature
        
        return output

