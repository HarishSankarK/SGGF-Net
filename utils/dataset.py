"""
Dataset loaders for VisDrone2021, AI-TOD, and HIT-UAV datasets
"""

import os
import xml.etree.ElementTree as ET
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class VisDroneDataset(Dataset):
    """
    VisDrone2021 Dataset Loader
    
    Classes: person, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, car
    """
    
    CLASSES = [
        'ignored', 'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
    ]
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of VisDrone dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # VisDrone structure: annotations/ and images/
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.annotation_dir = os.path.join(root_dir, 'annotations', split)
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) 
            if f.lower().endswith(('.jpg', '.png'))
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load annotations
        ann_file = self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt')
        ann_path = os.path.join(self.annotation_dir, ann_file)
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                    parts = line.split(',')
                    if len(parts) >= 6:
                        x1 = float(parts[0])
                        y1 = float(parts[1])
                        w = float(parts[2])
                        h = float(parts[3])
                        category = int(parts[5])
                        
                        # Filter out ignored regions (category 0)
                        if category > 0 and category < len(self.CLASSES):
                            # Convert to center format: [x_center, y_center, width, height]
                            x_center = x1 + w / 2.0
                            y_center = y1 + h / 2.0
                            
                            boxes.append([x_center, y_center, w, h])
                            labels.append(category)
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Convert image to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target


class AITODDataset(Dataset):
    """
    AI-TOD Dataset Loader for tiny object detection
    
    Classes: airplane, bridge, oil-tank, boat, swimming-pool, vehicle, person, windmill
    """
    
    CLASSES = [
        'background', 'airplane', 'bridge', 'oil-tank', 'boat',
        'swimming-pool', 'vehicle', 'person', 'windmill'
    ]
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of AI-TOD dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.annotation_dir = os.path.join(root_dir, 'annotations', split)
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load annotations (assuming XML format similar to PASCAL VOC)
        ann_file = self.image_files[idx].replace('.jpg', '.xml').replace('.png', '.xml')
        ann_path = os.path.join(self.annotation_dir, ann_file)
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in self.CLASSES:
                    label = self.CLASSES.index(class_name)
                    
                    bbox = obj.find('bndbox')
                    x1 = float(bbox.find('xmin').text)
                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    
                    # Convert to center format
                    w = x2 - x1
                    h = y2 - y1
                    x_center = x1 + w / 2.0
                    y_center = y1 + h / 2.0
                    
                    boxes.append([x_center, y_center, w, h])
                    labels.append(label)
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Convert image to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target


class HITUAVDataset(Dataset):
    """
    HIT-UAV Dataset Loader for infrared thermal object detection
    
    HIT-UAV is a high-altitude infrared thermal dataset with 2,898 images.
    Classes: Person, Car, Bicycle, OtherVehicle, DontCare
    
    Reference: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
    """
    
    CLASSES = [
        'background', 'Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare'
    ]
    
    def __init__(self, root_dir, split='train', transform=None, convert_to_rgb=True):
        """
        Args:
            root_dir: Root directory of HIT-UAV dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
            convert_to_rgb: If True, converts grayscale infrared to RGB (3 channels)
                           If False, keeps as single channel (requires model modification)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.convert_to_rgb = convert_to_rgb
        
        # HIT-UAV structure may vary, try common structures
        possible_image_dirs = [
            os.path.join(root_dir, 'images', split),
            os.path.join(root_dir, split, 'images'),
            os.path.join(root_dir, split),
        ]
        
        self.image_dir = None
        for img_dir in possible_image_dirs:
            if os.path.exists(img_dir):
                self.image_dir = img_dir
                break
        
        if self.image_dir is None:
            raise ValueError(f"Could not find image directory for split '{split}' in {root_dir}")
        
        # Annotation directory (HIT-UAV uses 'labels' folder)
        possible_ann_dirs = [
            os.path.join(root_dir, 'labels', split),
            os.path.join(root_dir, 'annotations', split),
            os.path.join(root_dir, split, 'labels'),
            os.path.join(root_dir, split, 'annotations'),
        ]
        
        self.annotation_dir = None
        for ann_dir in possible_ann_dirs:
            if os.path.exists(ann_dir):
                self.annotation_dir = ann_dir
                break
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image (infrared thermal - typically grayscale)
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path)
        
        # Convert to RGB if needed (for compatibility with 3-channel models)
        if self.convert_to_rgb:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            # Keep as grayscale (single channel)
            if image.mode != 'L':
                image = image.convert('L')
        
        image = np.array(image)
        
        # Load annotations
        # HIT-UAV may use JSON, XML, or TXT format
        boxes = []
        labels = []
        
        # Try different annotation formats
        base_name = os.path.splitext(self.image_files[idx])[0]
        
        # Try JSON format
        json_path = os.path.join(self.annotation_dir, base_name + '.json') if self.annotation_dir else None
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                for obj in data.get('objects', []):
                    bbox = obj.get('bbox', {})
                    class_name = obj.get('category', '')
                    
                    if class_name in self.CLASSES:
                        label = self.CLASSES.index(class_name)
                        x1 = bbox.get('x1', 0)
                        y1 = bbox.get('y1', 0)
                        x2 = bbox.get('x2', 0)
                        y2 = bbox.get('y2', 0)
                        
                        w = x2 - x1
                        h = y2 - y1
                        x_center = x1 + w / 2.0
                        y_center = y1 + h / 2.0
                        
                        boxes.append([x_center, y_center, w, h])
                        labels.append(label)
        
        # Try XML format (PASCAL VOC style)
        xml_path = os.path.join(self.annotation_dir, base_name + '.xml') if self.annotation_dir else None
        if xml_path and os.path.exists(xml_path) and not boxes:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in self.CLASSES:
                        label = self.CLASSES.index(class_name)
                        
                        bbox = obj.find('bndbox')
                        x1 = float(bbox.find('xmin').text)
                        y1 = float(bbox.find('ymin').text)
                        x2 = float(bbox.find('xmax').text)
                        y2 = float(bbox.find('ymax').text)
                        
                        w = x2 - x1
                        h = y2 - y1
                        x_center = x1 + w / 2.0
                        y_center = y1 + h / 2.0
                        
                        boxes.append([x_center, y_center, w, h])
                        labels.append(label)
            except:
                pass
        
        # Try TXT format (YOLO style - HIT-UAV uses this format)
        txt_path = os.path.join(self.annotation_dir, base_name + '.txt') if self.annotation_dir else None
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        # YOLO format: class_id x_center y_center width height (normalized)
                        try:
                            class_id = int(parts[0])
                            x_center_norm = float(parts[1])
                            y_center_norm = float(parts[2])
                            w_norm = float(parts[3])
                            h_norm = float(parts[4])
                            
                            # Convert normalized coordinates to absolute
                            if len(image.shape) == 2:  # Grayscale
                                img_h, img_w = image.shape
                            else:  # RGB or other
                                img_h, img_w = image.shape[:2]
                            
                            x_center = x_center_norm * img_w
                            y_center = y_center_norm * img_h
                            w = w_norm * img_w
                            h = h_norm * img_h
                            
                            # Skip DontCare class (class_id 4) or map it appropriately
                            # Map class_id to our CLASSES (0=background, 1=Person, 2=Car, 3=Bicycle, 4=OtherVehicle)
                            # HIT-UAV: 0=Person, 1=Car, 2=Bicycle, 3=OtherVehicle, 4=DontCare
                            # We map: HIT-UAV 0->1 (Person), 1->2 (Car), 2->3 (Bicycle), 3->4 (OtherVehicle), skip 4 (DontCare)
                            if class_id < 4:  # Skip DontCare (class 4)
                                mapped_class_id = class_id + 1  # Map to our class indices (0=bg, 1=Person, etc.)
                                if mapped_class_id < len(self.CLASSES):
                                    boxes.append([x_center, y_center, w, h])
                                    labels.append(mapped_class_id)
                        except Exception as e:
                            pass
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Convert image to tensor
        if len(image.shape) == 2:  # Grayscale
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        else:  # RGB
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target
