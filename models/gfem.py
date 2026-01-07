"""
Global Feature Extraction Module (GFEM)
Uses self-attention mechanism to capture long-range dependencies in images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C) where B=batch, N=num_patches, C=embed_dim
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        Q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        K = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, N, D)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, N, N)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (B, H, N, D)
        attended = attended.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        
        # Output projection
        output = self.out_proj(attended)
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class GFEM(nn.Module):
    """
    Global Feature Extraction Module
    
    Uses self-attention mechanism to capture and integrate long-range dependencies
    within images, optimizing feature extraction from a global perspective.
    """
    
    def __init__(self, in_channels=3, embed_dim=256, patch_size=4, num_layers=4, 
                 num_heads=8, mlp_ratio=4, dropout=0.1):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension for patches
            patch_size: Size of each patch (4x4 as per paper)
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            dropout: Dropout rate
        """
        super(GFEM, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch projection: divides image into non-overlapping patches
        # Using conv2d with kernel_size=patch_size and stride=patch_size
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            padding=0
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape (B, C, H, W)
        Returns:
            Feature maps of shape (B, embed_dim, H', W') where H'=H/patch_size, W'=W/patch_size
        """
        B, C, H, W = x.shape
        
        # Patch projection: (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.patch_proj(x)  # (B, embed_dim, H', W')
        
        # Reshape for transformer: (B, embed_dim, H', W') -> (B, H'*W', embed_dim)
        B, C, H_patch, W_patch = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim) where N = H' * W'
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial format: (B, N, embed_dim) -> (B, embed_dim, H', W')
        x = x.transpose(1, 2).view(B, self.embed_dim, H_patch, W_patch)
        
        return x

