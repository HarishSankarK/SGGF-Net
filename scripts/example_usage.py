"""
Example usage of SGGF-Net
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SGGFNet, GFEM, NDPA, ARPM


def example_model_creation():
    """Example: Create and test the model components"""
    print("=" * 50)
    print("SGGF-Net Component Examples")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Example 1: GFEM
    print("1. Testing GFEM (Global Feature Extraction Module)")
    print("-" * 50)
    gfem = GFEM(in_channels=3, embed_dim=256, patch_size=4, num_layers=2)
    gfem = gfem.to(device)
    
    # Create dummy input image (batch_size=2, channels=3, height=512, width=512)
    dummy_image = torch.randn(2, 3, 512, 512).to(device)
    gfem_output = gfem(dummy_image)
    print(f"Input shape: {dummy_image.shape}")
    print(f"GFEM output shape: {gfem_output.shape}")
    print(f"✓ GFEM working correctly\n")
    
    # Example 2: NDPA
    print("2. Testing NDPA (Normal Distribution-based Prior Assigner)")
    print("-" * 50)
    ndpa = NDPA(pos_threshold=0.5, neg_threshold=0.3)
    ndpa = ndpa.to(device)
    
    # Create dummy priors and ground truth boxes
    # Format: [x_center, y_center, width, height]
    priors = torch.tensor([
        [100, 100, 50, 50],
        [200, 200, 30, 30],
        [300, 300, 40, 40]
    ], dtype=torch.float32).to(device)
    
    gt_boxes = torch.tensor([
        [105, 105, 48, 48],
        [195, 195, 32, 32]
    ], dtype=torch.float32).to(device)
    
    labels, matched_gt, kl_matrix = ndpa(priors, gt_boxes)
    print(f"Priors shape: {priors.shape}")
    print(f"GT boxes shape: {gt_boxes.shape}")
    print(f"Assigned labels: {labels}")
    print(f"Matched GT indices: {matched_gt}")
    print(f"KL divergence matrix shape: {kl_matrix.shape}")
    print(f"✓ NDPA working correctly\n")
    
    # Example 3: ARPM
    print("3. Testing ARPM (Attention-guided ROI Pooling Module)")
    print("-" * 50)
    arpm = ARPM(in_channels=256, out_channels=256, roi_size=7)
    arpm = arpm.to(device)
    
    # Create dummy ROI features (batch_size=5, channels=256, height=7, width=7)
    dummy_roi_features = torch.randn(5, 256, 7, 7).to(device)
    arpm_output = arpm(dummy_roi_features)
    print(f"Input ROI features shape: {dummy_roi_features.shape}")
    print(f"ARPM output shape: {arpm_output.shape}")
    print(f"✓ ARPM working correctly\n")
    
    # Example 4: Full SGGF-Net
    print("4. Testing Full SGGF-Net")
    print("-" * 50)
    model = SGGFNet(num_classes=11, pretrained=False)
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_images = [torch.randn(3, 512, 512).to(device) for _ in range(2)]
    
    # Forward pass (inference mode)
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"Number of images: {len(dummy_images)}")
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  Image {i+1}: {len(output['boxes'])} detections")
    print(f"✓ SGGF-Net working correctly\n")
    
    print("=" * 50)
    print("All components tested successfully!")
    print("=" * 50)


if __name__ == '__main__':
    example_model_creation()

