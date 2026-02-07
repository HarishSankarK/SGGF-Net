"""
Path Aggregation Network (PANet)
Enhances FPN with bottom-up path augmentation for better feature propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet)
    
    PANet enhances FPN by adding a bottom-up path augmentation.
    It has both top-down (FPN-like) and bottom-up paths with lateral connections.
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: List of input channel numbers for each level [C2, C3, C4, C5]
            out_channels: Output channel number (default: 256)
        """
        super(PANet, self).__init__()
        self.out_channels = out_channels
        
        # Lateral connections (for top-down path, same as FPN)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Top-down path output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Bottom-up path convolutions (PANet addition)
        self.bottom_up_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # Final output convolutions (after bottom-up path)
        self.panet_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from backbone [C2, C3, C4, C5]
                    Each has shape [B, C_i, H_i, W_i]
        Returns:
            List of enhanced feature maps [P2, P3, P4, P5, P6]
        """
        # Step 1: Top-down path (FPN-like)
        # Apply lateral connections
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down features (from high to low resolution)
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample higher level feature and add to current level
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='nearest'
            )
        
        # Apply FPN output convolutions
        fpn_outputs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        # Step 2: Bottom-up path (PANet addition)
        # Start from P2 (lowest level)
        panet_features = [fpn_outputs[0]]  # P2
        
        # Build bottom-up features (from low to high resolution)
        for i in range(len(fpn_outputs) - 1):
            # Downsample current level and add to next level
            downsampled = self.bottom_up_convs[i](panet_features[i])
            # Add to next FPN output
            next_feat = fpn_outputs[i + 1] + F.interpolate(
                downsampled,
                size=fpn_outputs[i + 1].shape[2:],
                mode='nearest'
            )
            panet_features.append(next_feat)
        
        # Apply final PANet output convolutions
        panet_outputs = [panet_conv(feat) for panet_conv, feat in zip(self.panet_convs, panet_features)]
        
        # Add P6 (for detection head compatibility)
        if len(panet_outputs) > 0:
            p6 = F.max_pool2d(panet_outputs[-1], kernel_size=1, stride=2)
            panet_outputs.append(p6)
        
        return panet_outputs
