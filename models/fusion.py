"""
Mid-Level Fusion Module for RGB-Thermal Feature Integration
Implements concatenation, weighted combination, and cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism to enhance complementary features
    between RGB and Thermal modalities
    """
    
    def __init__(self, in_channels, reduction=8):
        super(CrossModalAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Channel attention for each modality
        self.channel_attention_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.channel_attention_thermal = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Cross-modal interaction
        self.cross_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat, thermal_feat):
        """
        Args:
            rgb_feat: RGB features [B, C, H, W]
            thermal_feat: Thermal features [B, C, H, W]
        Returns:
            Enhanced RGB and Thermal features
        """
        # Channel attention for each modality
        rgb_att = self.channel_attention_rgb(rgb_feat)
        thermal_att = self.channel_attention_thermal(thermal_feat)
        
        # Apply channel attention
        rgb_enhanced = rgb_feat * rgb_att
        thermal_enhanced = thermal_feat * thermal_att
        
        # Cross-modal attention
        concat_feat = torch.cat([rgb_enhanced, thermal_enhanced], dim=1)
        cross_att = self.cross_attention(concat_feat)
        
        # Cross-modal enhancement
        rgb_cross = rgb_enhanced * cross_att
        thermal_cross = thermal_enhanced * cross_att
        
        return rgb_cross, thermal_cross


class MidLevelFusion(nn.Module):
    """
    Mid-Level Fusion Module
    Combines RGB and Thermal features using multiple fusion strategies
    """
    
    def __init__(self, in_channels=256, fusion_type='concat_attention'):
        """
        Args:
            in_channels: Number of input channels for each modality
            fusion_type: 'concat', 'weighted', 'attention', or 'concat_attention'
        """
        super(MidLevelFusion, self).__init__()
        self.fusion_type = fusion_type
        self.in_channels = in_channels
        
        if fusion_type == 'concat':
            # Simple concatenation
            self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            
        elif fusion_type == 'weighted':
            # Learnable weighted combination
            self.weight_rgb = nn.Parameter(torch.ones(1))
            self.weight_thermal = nn.Parameter(torch.ones(1))
            self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            
        elif fusion_type == 'attention':
            # Cross-modal attention only
            self.cross_attention = CrossModalAttention(in_channels)
            self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            
        elif fusion_type == 'concat_attention':
            # Concatenation + Cross-modal attention (recommended)
            self.cross_attention = CrossModalAttention(in_channels)
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, rgb_feat, thermal_feat):
        """
        Args:
            rgb_feat: RGB features [B, C, H, W]
            thermal_feat: Thermal features [B, C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        if self.fusion_type == 'concat':
            # Simple concatenation
            fused = torch.cat([rgb_feat, thermal_feat], dim=1)
            fused = self.fusion_conv(fused)
            
        elif self.fusion_type == 'weighted':
            # Weighted combination
            weight_rgb_norm = torch.sigmoid(self.weight_rgb)
            weight_thermal_norm = torch.sigmoid(self.weight_thermal)
            weighted_rgb = rgb_feat * weight_rgb_norm
            weighted_thermal = thermal_feat * weight_thermal_norm
            fused = torch.cat([weighted_rgb, weighted_thermal], dim=1)
            fused = self.fusion_conv(fused)
            
        elif self.fusion_type == 'attention':
            # Cross-modal attention
            rgb_enhanced, thermal_enhanced = self.cross_attention(rgb_feat, thermal_feat)
            fused = torch.cat([rgb_enhanced, thermal_enhanced], dim=1)
            fused = self.fusion_conv(fused)
            
        elif self.fusion_type == 'concat_attention':
            # Concatenation + Cross-modal attention (best performance)
            rgb_enhanced, thermal_enhanced = self.cross_attention(rgb_feat, thermal_feat)
            fused = torch.cat([rgb_enhanced, thermal_enhanced], dim=1)
            fused = self.fusion_conv(fused)
            
        return fused
