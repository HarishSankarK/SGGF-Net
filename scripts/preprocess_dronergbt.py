"""
Preprocessing Script for DroneRGBT Dataset
Converts XML point annotations to YOLO format bounding boxes
Organizes dataset structure for training
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path


def point_to_bbox(x, y, img_width, img_height, bbox_size=32):
    """
    Convert point annotation to bounding box
    
    Args:
        x, y: Point coordinates
        img_width, img_height: Image dimensions
        bbox_size: Size of bounding box around point (default: 32 pixels)
    Returns:
        bbox: [x1, y1, x2, y2] in pixel coordinates
    """
    half_size = bbox_size // 2
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(img_width, x + half_size)
    y2 = min(img_height, y + half_size)
    
    return [x1, y1, x2, y2]


def xml_to_yolo(xml_path, class_mapping, bbox_size=32):
    """
    Convert XML annotation to YOLO format
    
    Args:
        xml_path: Path to XML file
        class_mapping: Dict mapping class names to class IDs
        bbox_size: Size of bounding box around points
    Returns:
        yolo_lines: List of YOLO format strings
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    yolo_lines = []
    
    # Process each object
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower()
        
        # Skip if class not in mapping
        if class_name not in class_mapping:
            continue
        
        class_id = class_mapping[class_name]
        
        # Check if point annotation exists
        point = obj.find('point')
        if point is not None:
            x = int(point.find('x').text)
            y = int(point.find('y').text)
            
            # Convert point to bounding box
            bbox = point_to_bbox(x, y, img_width, img_height, bbox_size)
            x1, y1, x2, y2 = bbox
            
            # Convert to YOLO format (normalized center_x, center_y, width, height)
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Check if bounding box exists (some datasets have both)
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Convert to YOLO format
            center_x = ((xmin + xmax) / 2) / img_width
            center_y = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    return yolo_lines


def organize_dronergbt_dataset(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, bbox_size=32, overwrite=False):
    """
    Organize DroneRGBT dataset for training
    
    Args:
        source_dir: Source directory (sggf_net/data/DroneRGBT)
        target_dir: Target directory (sggf_net/data/dronergbt)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        bbox_size: Size of bounding boxes around points
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Check if target directory already has preprocessed data
    if target_dir.exists() and not overwrite:
        existing_files = list((target_dir / 'rgb' / 'train').glob('*.jpg')) if (target_dir / 'rgb' / 'train').exists() else []
        if existing_files:
            print(f"⚠️  WARNING: Target directory already contains {len(existing_files)} preprocessed images!")
            print(f"   Target: {target_dir}")
            print(f"   Use --overwrite flag to overwrite existing data.")
            print(f"   Preprocessing cancelled.")
            return
    
    # Class mapping (DroneRGBT - actual dataset only contains "person")
    # Note: The dataset loader will remap class 0 (person) to class 1 in the final system
    class_mapping = {
        'person': 0,  # Only class present in actual dataset
        # Other classes defined but not present in actual data:
        # 'car': 1,
        # 'bicycle': 2,
        # 'motorcycle': 3,
        # 'bus': 4,
        # 'truck': 5,
        # 'van': 6,
        # 'dog': 7,
        # 'cat': 8
    }
    
    # Create target directory structure
    for split in ['train', 'val', 'test']:
        (target_dir / 'rgb' / split).mkdir(parents=True, exist_ok=True)
        (target_dir / 'thermal' / split).mkdir(parents=True, exist_ok=True)
        (target_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)
    
    # Process Train folder
    train_rgb_dir = source_dir / 'Train' / 'RGB'
    train_infrared_dir = source_dir / 'Train' / 'Infrared'
    train_gt_dir = source_dir / 'Train' / 'GT_'
    
    # Get all RGB images
    rgb_images = sorted([f for f in train_rgb_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    print(f"Found {len(rgb_images)} training images")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(rgb_images)
    
    n_train = int(len(rgb_images) * train_ratio)
    n_val = int(len(rgb_images) * val_ratio)
    
    train_images = rgb_images[:n_train]
    val_images = rgb_images[n_train:n_train + n_val]
    test_images = rgb_images[n_train + n_val:]
    
    print(f"Split: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
    
    # Process each split
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split_name, images in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        for rgb_img_path in images:
            rgb_img_name = rgb_img_path.name
            base_name = rgb_img_path.stem  # e.g., "1" from "1.jpg"
            
            # Find corresponding infrared image (with R suffix)
            infrared_img_name = f"{base_name}R.jpg"
            infrared_img_path = train_infrared_dir / infrared_img_name
            
            # If not found, try other extensions
            if not infrared_img_path.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    infrared_img_path = train_infrared_dir / f"{base_name}R{ext}"
                    if infrared_img_path.exists():
                        break
            
            # Find corresponding XML annotation
            xml_name = f"{base_name}R.xml"
            xml_path = train_gt_dir / xml_name
            
            if not xml_path.exists():
                print(f"Warning: XML not found for {rgb_img_name}, skipping...")
                continue
            
            if not infrared_img_path.exists():
                print(f"Warning: Infrared image not found for {rgb_img_name}, skipping...")
                continue
            
            # Copy RGB image
            shutil.copy2(rgb_img_path, target_dir / 'rgb' / split_name / rgb_img_name)
            
            # Copy infrared image (rename to match RGB for consistency)
            shutil.copy2(infrared_img_path, target_dir / 'thermal' / split_name / rgb_img_name)
            
            # Convert XML to YOLO format
            yolo_lines = xml_to_yolo(xml_path, class_mapping, bbox_size)
            
            # Save YOLO annotation
            yolo_path = target_dir / 'annotations' / split_name / f"{base_name}.txt"
            with open(yolo_path, 'w') as f:
                f.writelines(yolo_lines)
        
        print(f"✓ {split_name}: {len(images)} images processed")
    
    # Process Test folder (if exists)
    test_rgb_dir = source_dir / 'Test' / 'RGB'
    test_infrared_dir = source_dir / 'Test' / 'Infrared'
    test_gt_dir = source_dir / 'Test' / 'GT_'
    
    if test_rgb_dir.exists():
        print(f"\nProcessing Test folder...")
        test_images = sorted([f for f in test_rgb_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        for rgb_img_path in test_images:
            rgb_img_name = rgb_img_path.name
            base_name = rgb_img_path.stem
            
            # Find corresponding files
            infrared_img_name = f"{base_name}R.jpg"
            infrared_img_path = test_infrared_dir / infrared_img_name
            
            if not infrared_img_path.exists():
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    infrared_img_path = test_infrared_dir / f"{base_name}R{ext}"
                    if infrared_img_path.exists():
                        break
            
            xml_name = f"{base_name}R.xml"
            xml_path = test_gt_dir / xml_name
            
            if not xml_path.exists() or not infrared_img_path.exists():
                continue
            
            # Copy files
            shutil.copy2(rgb_img_path, target_dir / 'rgb' / 'test' / rgb_img_name)
            shutil.copy2(infrared_img_path, target_dir / 'thermal' / 'test' / rgb_img_name)
            
            # Convert and save annotation
            yolo_lines = xml_to_yolo(xml_path, class_mapping, bbox_size)
            yolo_path = target_dir / 'annotations' / 'test' / f"{base_name}.txt"
            with open(yolo_path, 'w') as f:
                f.writelines(yolo_lines)
        
        print(f"✓ Test: {len(test_images)} images processed")
    
    # Create dataset info file
    info_path = target_dir / 'dataset_info.txt'
    with open(info_path, 'w') as f:
        f.write("DroneRGBT Dataset - Preprocessed\n")
        f.write("=" * 50 + "\n\n")
        f.write("ACTUAL CLASSES IN DATASET:\n")
        f.write("  0: person (ONLY class present in actual annotations)\n\n")
        f.write("NOTE: Preprocessing script assumes 9 classes, but actual XML/annotations only contain 'person'.\n")
        f.write("The following classes are defined in preprocessing but NOT present in actual data:\n")
        f.write("  1: car\n")
        f.write("  2: bicycle\n")
        f.write("  3: motorcycle\n")
        f.write("  4: bus\n")
        f.write("  5: truck\n")
        f.write("  6: van\n")
        f.write("  7: dog\n")
        f.write("  8: cat\n\n")
        f.write("SINGLE-CLASS SYSTEM (PERSON ONLY):\n")
        f.write("  0: background\n")
        f.write("  1: person (from original class 0, remapped by dataset loader)\n\n")
        f.write(f"Bounding Box Size: {bbox_size} pixels\n")
        f.write(f"\nSplits:\n")
        for split in ['train', 'val', 'test']:
            count = len(list((target_dir / 'rgb' / split).glob('*.jpg')))
            f.write(f"  {split}: {count} images\n")
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Dataset organized in: {target_dir}")
    print(f"📊 Check {info_path} for dataset information")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess DroneRGBT dataset')
    parser.add_argument('--source_dir', type=str, 
                       default='sggf_net/data/DroneRGBT',
                       help='Source DroneRGBT directory')
    parser.add_argument('--target_dir', type=str,
                       default='sggf_net/data/dronergbt',
                       help='Target organized directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation data ratio')
    parser.add_argument('--bbox_size', type=int, default=32,
                       help='Bounding box size around points (pixels)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing preprocessed data')
    
    args = parser.parse_args()
    
    organize_dronergbt_dataset(
        args.source_dir,
        args.target_dir,
        args.train_ratio,
        args.val_ratio,
        args.bbox_size,
        args.overwrite
    )


if __name__ == '__main__':
    main()
