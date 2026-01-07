"""
Quick test script to verify HIT-UAV dataset loading
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import HITUAVDataset, get_train_transform

def test_dataset():
    """Test HIT-UAV dataset loading"""
    # Try data/hit-uav first (Colab structure), then ../hit-uav (local)
    import os
    if os.path.exists('data/hit-uav'):
        data_dir = 'data/hit-uav'
    elif os.path.exists('../hit-uav'):
        data_dir = '../hit-uav'
    else:
        data_dir = 'data/hit-uav'  # Default for Colab
    
    print("Testing HIT-UAV Dataset Loader...")
    print(f"Data directory: {data_dir}")
    print("-" * 50)
    
    # Test without transforms first
    try:
        dataset = HITUAVDataset(data_dir, split='train', transform=None, convert_to_rgb=True)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total images: {len(dataset)}")
        
        # Test loading one sample
        if len(dataset) > 0:
            image, target = dataset[0]
            print(f"✓ Sample loaded successfully!")
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Number of boxes: {len(target['boxes'])}")
            print(f"  Number of labels: {len(target['labels'])}")
            if len(target['boxes']) > 0:
                print(f"  First box: {target['boxes'][0]}")
                print(f"  First label: {target['labels'][0]}")
        
        # Test with transforms
        print("\nTesting with transforms...")
        transform = get_train_transform(max_size=1536)
        dataset_transformed = HITUAVDataset(data_dir, split='train', transform=transform, convert_to_rgb=True)
        image, target = dataset_transformed[0]
        print(f"✓ Transformed sample loaded!")
        print(f"  Transformed image shape: {image.shape}")
        
        print("\n" + "=" * 50)
        print("Dataset test PASSED! Ready for training.")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_dataset()

