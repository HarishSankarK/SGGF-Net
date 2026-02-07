"""
Preprocessing Script for SMOD Dataset (Archive Format)
Converts COCO format JSON annotations to YOLO format
Organizes dataset structure for training
"""

import os
import json
import shutil
import random
from pathlib import Path


def coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO bbox format [x, y, width, height] to YOLO format [center_x, center_y, width, height] (normalized)
    
    Args:
        bbox: [x, y, width, height] in pixel coordinates
        img_width, img_height: Image dimensions
    Returns:
        center_x, center_y, width, height: Normalized (0.0-1.0)
    """
    x, y, w, h = bbox
    
    # Convert to center coordinates
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    # Clamp to [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height


def preprocess_smod_archive(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Preprocess SMOD dataset from archive format (COCO JSON) to YOLO format
    
    Args:
        source_dir: Source directory (sggf_net/data/archive)
        target_dir: Target directory (sggf_net/data/SMOD)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Category mapping: COCO category_id -> YOLO class_id
    # COCO: 1=person, 2=rider, 3=bicycle, 4=car
    # YOLO: 0=person, 1=rider, 2=bicycle, 3=car
    category_to_class = {
        1: 0,  # person
        2: 1,  # rider
        3: 2,  # bicycle
        4: 3   # car
    }
    
    # Create target directory structure
    for split in ['train', 'val', 'test']:
        (target_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process train annotations
    train_json_path = source_dir / 'anno' / 'new_train_annotations_rgb.json'
    test_json_path = source_dir / 'anno' / 'new_test_annotations_rgb.json'
    
    if not train_json_path.exists():
        raise ValueError(f"Train annotations not found: {train_json_path}")
    
    # Load train annotations
    print("Loading train annotations...")
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    # Create image_id to image info mapping
    image_dict = {img['id']: img for img in train_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in train_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Get all image IDs (including images without annotations)
    all_image_ids = set(image_dict.keys())
    annotated_image_ids = set(annotations_by_image.keys())
    
    # Use all images, but prioritize those with annotations
    image_ids = list(annotations_by_image.keys()) + list(all_image_ids - annotated_image_ids)
    random.seed(42)
    random.shuffle(image_ids)
    
    # Split into train/val/test
    n_train = int(len(image_ids) * train_ratio)
    n_val = int(len(image_ids) * val_ratio)
    
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    
    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Process each split
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, image_id_list in splits.items():
        print(f"\nProcessing {split_name} split...")
        processed = 0
        
        for image_id in image_id_list:
            img_info = image_dict[image_id]
            file_name = img_info['file_name']  # e.g., "day/000000_rgb.jpg"
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Find source image path
            source_img_path = source_dir / file_name
            if not source_img_path.exists():
                # Try without _rgb suffix
                base_name = file_name.replace('_rgb.jpg', '.jpg')
                source_img_path = source_dir / base_name
                if not source_img_path.exists():
                    continue
            
            # Get base name for target files (remove day/ or night/ prefix and _rgb suffix)
            # file_name is like "day/000000_rgb.jpg" or "night/000000_rgb.jpg"
            base_name = Path(file_name).stem.replace('_rgb', '')
            # Use a unique name to avoid conflicts between day and night images
            folder_name = Path(file_name).parent.name  # "day" or "night"
            unique_base_name = f"{folder_name}_{base_name}"
            
            # Copy image
            target_img_path = target_dir / 'images' / split_name / f"{unique_base_name}.jpg"
            shutil.copy2(source_img_path, target_img_path)
            
            # Convert annotations to YOLO format
            yolo_lines = []
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    category_id = ann['category_id']
                    if category_id not in category_to_class:
                        continue
                    
                    class_id = category_to_class[category_id]
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # Convert to YOLO format
                    center_x, center_y, width, height = coco_to_yolo_bbox(bbox, img_width, img_height)
                    
                    yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            # Save YOLO annotation
            target_label_path = target_dir / 'labels' / split_name / f"{unique_base_name}.txt"
            with open(target_label_path, 'w') as f:
                f.writelines(yolo_lines)
            
            processed += 1
        
        print(f"✓ {split_name}: {processed} images processed")
    
    # Process test annotations if available
    if test_json_path.exists():
        print(f"\nProcessing test annotations...")
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)
        
        test_image_dict = {img['id']: img for img in test_data['images']}
        test_annotations_by_image = {}
        for ann in test_data['annotations']:
            image_id = ann['image_id']
            if image_id not in test_annotations_by_image:
                test_annotations_by_image[image_id] = []
            test_annotations_by_image[image_id].append(ann)
        
        processed = 0
        for image_id, img_info in test_image_dict.items():
            file_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            source_img_path = source_dir / file_name
            if not source_img_path.exists():
                base_name = file_name.replace('_rgb.jpg', '.jpg')
                source_img_path = source_dir / base_name
                if not source_img_path.exists():
                    continue
            
            base_name = Path(file_name).stem.replace('_rgb', '')
            folder_name = Path(file_name).parent.name
            unique_base_name = f"{folder_name}_{base_name}"
            
            # Copy to test split
            target_img_path = target_dir / 'images' / 'test' / f"{unique_base_name}.jpg"
            shutil.copy2(source_img_path, target_img_path)
            
            # Convert annotations
            yolo_lines = []
            if image_id in test_annotations_by_image:
                for ann in test_annotations_by_image[image_id]:
                    category_id = ann['category_id']
                    if category_id not in category_to_class:
                        continue
                    
                    class_id = category_to_class[category_id]
                    bbox = ann['bbox']
                    center_x, center_y, width, height = coco_to_yolo_bbox(bbox, img_width, img_height)
                    yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            target_label_path = target_dir / 'labels' / 'test' / f"{unique_base_name}.txt"
            with open(target_label_path, 'w') as f:
                f.writelines(yolo_lines)
            
            processed += 1
        
        print(f"✓ Test: {processed} images processed")
    
    # Create dataset info file
    info_path = target_dir / 'dataset_info.txt'
    with open(info_path, 'w') as f:
        f.write("SMOD Dataset - Preprocessed\n")
        f.write("=" * 50 + "\n\n")
        f.write("Original Classes (COCO format):\n")
        f.write("  1: person\n")
        f.write("  2: rider\n")
        f.write("  3: bicycle\n")
        f.write("  4: car\n\n")
        f.write("YOLO Format Classes:\n")
        f.write("  0: person\n")
        f.write("  1: rider\n")
        f.write("  2: bicycle\n")
        f.write("  3: car\n\n")
        f.write("2-Class System (after remapping in dataset loader):\n")
        f.write("  0: background\n")
        f.write("  1: person (from classes 0 and 1)\n")
        f.write("  2: vehicle (from classes 2 and 3)\n\n")
        f.write("Splits:\n")
        for split in ['train', 'val', 'test']:
            count = len(list((target_dir / 'images' / split).glob('*.jpg')))
            f.write(f"  {split}: {count} images\n")
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📁 Dataset organized in: {target_dir}")
    print(f"📊 Check {info_path} for dataset information")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess SMOD dataset from archive format')
    parser.add_argument('--source_dir', type=str,
                       default='sggf_net/data/archive',
                       help='Source archive directory')
    parser.add_argument('--target_dir', type=str,
                       default='sggf_net/data/SMOD',
                       help='Target organized directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation data ratio')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing preprocessed data')
    
    args = parser.parse_args()
    
    # Check if target directory already has data
    if not args.overwrite and (Path(args.target_dir) / 'images' / 'train').exists():
        existing_files = list((Path(args.target_dir) / 'images' / 'train').glob('*.jpg'))
        if existing_files:
            print(f"⚠️  WARNING: Target directory already contains {len(existing_files)} preprocessed images!")
            print(f"   Target: {args.target_dir}")
            print(f"   Use --overwrite flag to overwrite existing data.")
            print(f"   Preprocessing cancelled.")
            return
    
    preprocess_smod_archive(
        args.source_dir,
        args.target_dir,
        args.train_ratio,
        args.val_ratio
    )


if __name__ == '__main__':
    main()
