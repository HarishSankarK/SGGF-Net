"""
Preprocessing Script for HIT-UAV Dataset
Converts raw HIT-UAV (JSON/XML or yolo_labels) to standard structure for Fusion-YOLOv11

Raw HIT-UAV from https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset has:
- normal_json, normal_xml: standard bbox annotations
- yolo_labels: YOLO format (if available)
- Images: typically in dataset root or subdirs

Output structure (expected by HITUAVDataset):
  target_dir/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/

HIT-UAV classes: 0=Person, 1=Car, 2=Bicycle, 3=OtherVehicle, 4=DontCare
We keep 0-3, skip 4 (DontCare). Class IDs in YOLO remain 0-3.
"""

import os
import json
import shutil
import random
from pathlib import Path


def find_images_and_labels(source_dir):
    """Find image and annotation directories in raw HIT-UAV structure."""
    source = Path(source_dir)
    img_dirs = []
    ann_dirs = []
    
    # Common raw HIT-UAV layouts
    candidates_img = [
        source / 'images',
        source / 'Images',
        source / 'train' / 'images',
        source,
    ]
    for d in candidates_img:
        if d.exists() and any(d.iterdir()):
            imgs = list(d.rglob('*.jpg')) + list(d.rglob('*.png')) + list(d.rglob('*.bmp'))
            if imgs:
                img_dirs.append(d)
                break
    
    # Look for images in root
    if not img_dirs:
        imgs = list(source.rglob('*.jpg')) + list(source.rglob('*.png')) + list(source.rglob('*.bmp'))
        imgs = [p for p in imgs if 'readme' not in str(p).lower() and 'anno' not in str(p).lower()]
        if imgs:
            img_dirs = [imgs[0].parent]
    
    # Annotations: yolo_labels, normal_json, normal_xml, labels
    candidates_ann = [
        source / 'yolo_labels',
        source / 'labels',
        source / 'normal_json',
        source / 'normal_xml',
    ]
    for d in candidates_ann:
        if d.exists():
            ann_dirs.append(d)
    
    return img_dirs, ann_dirs


def json_to_yolo_bbox(bbox, img_w, img_h):
    """Convert JSON bbox to YOLO normalized [center_x, center_y, width, height]."""
    if len(bbox) < 4:
        return None
    # Already normalized (YOLO format)
    if all(0 <= v <= 1 for v in bbox[:4]):
        return tuple(bbox[:4])
    # [x1, y1, x2, y2] in pixels
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return (xc, yc, w, h)
    # [xc, yc, w, h] in pixels (HIT-UAV standard format)
    xc, yc, w, h = bbox[0] / img_w, bbox[1] / img_h, bbox[2] / img_w, bbox[3] / img_h
    return (xc, yc, w, h)


def preprocess_hituav(source_dir, target_dir, train_ratio=0.85, seed=42, overwrite=False):
    """
    Preprocess HIT-UAV dataset to standard structure.
    
    Args:
        source_dir: Raw HIT-UAV root (with images, yolo_labels or normal_json)
        target_dir: Output dir (images/train, labels/train, etc.)
        train_ratio: Fraction for train split (default 0.85)
        overwrite: If False, skip when target already has structure
    """
    source = Path(source_dir)
    target = Path(target_dir)
    
    # If target already has correct structure, no preprocessing needed
    if (target / 'images' / 'train').exists() and (target / 'labels' / 'train').exists() and not overwrite:
        n_tr = len(list((target / 'images' / 'train').glob('*.*')))
        n_val = len(list((target / 'images' / 'val').glob('*.*'))) if (target / 'images' / 'val').exists() else 0
        print(f"HIT-UAV already preprocessed at {target} ({n_tr} train, {n_val} val). Use --overwrite to re-run.")
        return n_tr, n_val
    
    img_dirs, ann_dirs = find_images_and_labels(source)
    if not img_dirs:
        raise ValueError(f"No images found in {source}. Expected: images/ or .jpg/.png in root.")
    
    img_root = img_dirs[0]
    
    # Collect all images (handle nested dirs)
    all_imgs = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        all_imgs.extend(img_root.rglob(ext))
    all_imgs = sorted(set(all_imgs))
    
    # Build image base_name -> path
    base_to_path = {}
    for p in all_imgs:
        base = p.stem
        if base not in base_to_path:
            base_to_path[base] = p
    
    # Find annotation format
    ann_root = None
    ann_format = None  # 'yolo', 'json', 'xml'
    
    for ad in ann_dirs:
        ad = Path(ad)
        sample = next(ad.rglob('*'), None)
        if sample:
            suffix = sample.suffix.lower()
            if suffix == '.txt':
                ann_format = 'yolo'
                ann_root = ad
                break
            elif suffix == '.json':
                ann_format = 'json'
                ann_root = ad
                break
            elif suffix == '.xml':
                ann_format = 'xml'
                ann_root = ad
                break
    
    if ann_root is None:
        raise ValueError(f"No annotations (txt/json/xml) found in {source}. Check yolo_labels, normal_json, or normal_xml.")
    
    # Create target structure
    for split in ['train', 'val']:
        (target / 'images' / split).mkdir(parents=True, exist_ok=True)
        (target / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    keys = list(base_to_path.keys())
    random.seed(seed)
    random.shuffle(keys)
    n_train = int(len(keys) * train_ratio)
    train_keys = set(keys[:n_train])
    
    # HIT-UAV class: 0=Person, 1=Car, 2=Bicycle, 3=OtherVehicle, 4=DontCare (skip)
    def process_ann(base_name, img_path, out_label_path):
        img = __import__('PIL.Image').Image.open(img_path)
        w, h = img.size
        
        lines = []
        ann_txt = next(ann_root.rglob(f'{base_name}.txt'), None)
        ann_json = next(ann_root.rglob(f'{base_name}.json'), None)
        ann_xml = next(ann_root.rglob(f'{base_name}.xml'), None)
        
        if ann_format == 'yolo' and ann_txt is not None and ann_txt.exists():
            with open(ann_txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) < 4:  # skip DontCare
                        lines.append(line.strip())
        elif ann_format == 'json' and ann_json is not None and ann_json.exists():
            with open(ann_json) as f:
                data = json.load(f)
            for obj in data.get('objects', data.get('annotations', data.get('shapes', []))):
                cat = obj.get('category', obj.get('category_id', obj.get('label', 0)))
                if isinstance(cat, str):
                    cat = {'Person': 0, 'Car': 1, 'Bicycle': 2, 'OtherVehicle': 3}.get(cat, 4)
                if cat >= 4:
                    continue
                bbox = obj.get('bbox', obj.get('box', []))
                if isinstance(bbox, dict):
                    if 'xmin' in bbox or 'xmax' in bbox:
                        bbox = [bbox.get('xmin', 0), bbox.get('ymin', 0),
                                bbox.get('xmax', 1), bbox.get('ymax', 1)]
                    else:
                        bbox = [bbox.get('x', 0), bbox.get('y', 0),
                                bbox.get('w', 1), bbox.get('h', 1)]
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    yo = json_to_yolo_bbox(bbox, w, h)
                    if yo:
                        lines.append(f"{cat} {yo[0]:.6f} {yo[1]:.6f} {yo[2]:.6f} {yo[3]:.6f}")
        elif ann_format == 'xml':
            import xml.etree.ElementTree as ET
            if ann_xml is not None and ann_xml.exists():
                try:
                    tree = ET.parse(ann_xml)
                    root = tree.getroot()
                    sz = root.find('size')
                    if sz is not None:
                        w = int(sz.find('width').text or w)
                        h = int(sz.find('height').text or h)
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        cat_map = {'Person': 0, 'Car': 1, 'Bicycle': 2, 'OtherVehicle': 3}
                        if name not in cat_map:
                            continue
                        cat = cat_map[name]
                        b = obj.find('bndbox')
                        if b is not None:
                            x1 = float(b.find('xmin').text)
                            y1 = float(b.find('ymin').text)
                            x2 = float(b.find('xmax').text)
                            y2 = float(b.find('ymax').text)
                            yo = json_to_yolo_bbox([x1, y1, x2, y2], w, h)
                            if yo:
                                lines.append(f"{cat} {yo[0]:.6f} {yo[1]:.6f} {yo[2]:.6f} {yo[3]:.6f}")
                except Exception:
                    pass
        
        with open(out_label_path, 'w') as f:
            f.write('\n'.join(lines))
    
    for base_name in keys:
        img_path = base_to_path[base_name]
        split = 'train' if base_name in train_keys else 'val'
        out_img = target / 'images' / split / img_path.name
        out_label = target / 'labels' / split / f'{base_name}.txt'
        shutil.copy2(img_path, out_img)
        process_ann(base_name, img_path, out_label)
    
    n_train = len(train_keys)
    n_val = len(keys) - n_train
    print(f"HIT-UAV preprocessed: {n_train} train, {n_val} val -> {target}")
    return n_train, n_val


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Preprocess HIT-UAV dataset')
    p.add_argument('--source', type=str, default='data/HIT-UAV-raw', help='Raw HIT-UAV root')
    p.add_argument('--target', type=str, default='data/HIT-UAV', help='Output directory')
    p.add_argument('--train_ratio', type=float, default=0.85)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing target')
    args = p.parse_args()
    preprocess_hituav(args.source, args.target, args.train_ratio, args.seed, args.overwrite)
