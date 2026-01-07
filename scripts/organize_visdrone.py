"""
Script to organize VisDrone dataset into the expected structure
Run this after downloading the VisDrone-DET dataset files
"""

import os
import shutil
import argparse
from pathlib import Path


def organize_visdrone_dataset(download_dir, output_dir):
    """
    Organize VisDrone dataset into the expected structure
    
    Args:
        download_dir: Directory where you extracted the downloaded VisDrone files
        output_dir: Output directory with organized structure
    """
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)
    
    # VisDrone dataset typically comes with these folders:
    # - VisDrone2019-DET-train/
    # - VisDrone2019-DET-val/
    # - VisDrone2019-DET-test-dev/ or VisDrone2019-DET-test-challenge/
    
    download_path = Path(download_dir)
    
    # Map VisDrone folder names to our splits
    folder_mapping = {
        'VisDrone2019-DET-train': 'train',
        'VisDrone2019-DET-val': 'val',
        'VisDrone2019-DET-test-dev': 'test',
        'VisDrone2019-DET-test-challenge': 'test'
    }
    
    for visdrone_folder, split in folder_mapping.items():
        source_folder = download_path / visdrone_folder
        
        if not source_folder.exists():
            print(f"Warning: {source_folder} not found. Skipping...")
            continue
        
        # Copy images
        images_source = source_folder / 'images'
        if images_source.exists():
            for img_file in images_source.glob('*.jpg'):
                shutil.copy2(img_file, output_dir / 'images' / split / img_file.name)
            print(f"Copied images from {visdrone_folder} to images/{split}/")
        
        # Copy annotations
        annotations_source = source_folder / 'annotations'
        if annotations_source.exists():
            for ann_file in annotations_source.glob('*.txt'):
                shutil.copy2(ann_file, output_dir / 'annotations' / split / ann_file.name)
            print(f"Copied annotations from {visdrone_folder} to annotations/{split}/")
    
    print(f"\nDataset organized successfully!")
    print(f"Output directory: {output_dir}")
    print(f"\nYou can now use this path in training:")
    print(f"  python scripts/train.py --dataset visdrone --data_dir {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize VisDrone dataset')
    parser.add_argument('--download_dir', type=str, required=True,
                        help='Directory where you extracted the downloaded VisDrone files')
    parser.add_argument('--output_dir', type=str, default='../visdrone2021',
                        help='Output directory for organized dataset')
    
    args = parser.parse_args()
    organize_visdrone_dataset(args.download_dir, args.output_dir)

