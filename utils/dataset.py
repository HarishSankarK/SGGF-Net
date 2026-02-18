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
                            x2, y2 = x1 + w, y1 + h
                            boxes.append([x1, y1, x2, y2])
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
                    boxes.append([x1, y1, x2, y2])
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
    Original classes: Person, Car, Bicycle, OtherVehicle, DontCare
    
    With use_person_vehicle=True (default): Maps to person + vehicle (unified with DroneRGBT, SMOD)
    - Person → person (1)
    - Car, Bicycle, OtherVehicle → vehicle (2)
    - DontCare → background (skipped)
    
    Reference: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
    """
    
    CLASSES = [
        'background', 'Person', 'Car', 'Bicycle', 'OtherVehicle', 'DontCare'
    ]
    
    # 2-class system when use_person_vehicle=True (background + person + vehicle)
    CLASSES_2 = ['background', 'person', 'vehicle']
    
    @staticmethod
    def remap_class(original_class_id, use_person_vehicle=True):
        """
        Remap HIT-UAV classes to person/vehicle for unified training.
        HIT-UAV: 0=Person, 1=Car, 2=Bicycle, 3=OtherVehicle, 4=DontCare
        Returns: 1=person, 2=vehicle, or None to skip (DontCare)
        """
        if not use_person_vehicle:
            if original_class_id < 4:  # Skip DontCare
                return original_class_id + 1  # 0->1, 1->2, 2->3, 3->4
            return None
        if original_class_id == 0:  # Person → person (1)
            return 1
        if original_class_id in (1, 2, 3):  # Car, Bicycle, OtherVehicle → vehicle (2)
            return 2
        return None  # DontCare → skip
    
    def __init__(self, root_dir, split='train', transform=None, convert_to_rgb=True, use_person_vehicle=False):
        """
        Args:
            root_dir: Root directory of HIT-UAV dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
            convert_to_rgb: If True, converts grayscale infrared to RGB (3 channels)
            use_person_vehicle: If True, map Person→person(1), Car/Bicycle/OtherVehicle→vehicle(2), DontCare→skip (for Fusion unified training)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.convert_to_rgb = convert_to_rgb
        self.use_person_vehicle = use_person_vehicle
        
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
        
        # HIT-UAV name to YOLO id: Person=0, Car=1, Bicycle=2, OtherVehicle=3, DontCare=4
        name_to_id = {'Person': 0, 'Car': 1, 'Bicycle': 2, 'OtherVehicle': 3, 'DontCare': 4}
        
        # Try JSON format
        json_path = os.path.join(self.annotation_dir, base_name + '.json') if self.annotation_dir else None
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                for obj in data.get('objects', []):
                    bbox = obj.get('bbox', {})
                    class_name = obj.get('category', '')
                    orig_id = name_to_id.get(class_name, 4)
                    mapped = self.remap_class(orig_id, self.use_person_vehicle)
                    if mapped is None:
                        continue
                    x1 = bbox.get('x1', 0)
                    y1 = bbox.get('y1', 0)
                    x2 = bbox.get('x2', 0)
                    y2 = bbox.get('y2', 0)
                    w = x2 - x1
                    h = y2 - y1
                    x_center = x1 + w / 2.0
                    y_center = y1 + h / 2.0
                    boxes.append([x_center, y_center, w, h])
                    labels.append(mapped)
        
        # Try XML format (PASCAL VOC style)
        xml_path = os.path.join(self.annotation_dir, base_name + '.xml') if self.annotation_dir else None
        if xml_path and os.path.exists(xml_path) and not boxes:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    orig_id = name_to_id.get(class_name, 4)
                    mapped = self.remap_class(orig_id, self.use_person_vehicle)
                    if mapped is None:
                        continue
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
                    labels.append(mapped)
            except:
                pass
        
        # Try TXT format (YOLO style - HIT-UAV uses this format)
        txt_path = os.path.join(self.annotation_dir, base_name + '.txt') if self.annotation_dir else None
        if txt_path and os.path.exists(txt_path) and not boxes:
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center_norm = float(parts[1])
                            y_center_norm = float(parts[2])
                            w_norm = float(parts[3])
                            h_norm = float(parts[4])
                            if len(image.shape) == 2:
                                img_h, img_w = image.shape
                            else:
                                img_h, img_w = image.shape[:2]
                            x_center = x_center_norm * img_w
                            y_center = y_center_norm * img_h
                            w = w_norm * img_w
                            h = h_norm * img_h
                            mapped = self.remap_class(class_id, self.use_person_vehicle)
                            if mapped is not None:
                                boxes.append([x_center, y_center, w, h])
                                labels.append(mapped)
                        except Exception:
                            pass
        
        # Convert [x_center, y_center, w, h] -> [x1, y1, x2, y2] for assign_targets and metrics
        if boxes:
            boxes_xyxy = []
            for b in boxes:
                cx, cy, w, h = b[0], b[1], b[2], b[3]
                boxes_xyxy.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            boxes = boxes_xyxy
        
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


class SMODDataset(Dataset):
    """
    SMOD (Small and Medium Object Detection) Dataset Loader for RGB images
    
    SMOD is a UAV-based RGB dataset for small and medium object detection.
    
    Original classes: person (0), rider (1), bicycle (2), car (3)
    Mapped to 2-class system:
    - person (0) and rider (1) → person (1)
    - bicycle (2) and car (3) → vehicle (2)
    """
    
    # 2-class system: background, person, vehicle
    CLASSES = [
        'background', 'person', 'vehicle'
    ]
    
    # Original SMOD classes
    ORIGINAL_CLASSES = [
        'person', 'rider', 'bicycle', 'car'
    ]
    
    @staticmethod
    def remap_class(original_class_id):
        """
        Remap original SMOD class IDs to 2-class system (person, vehicle)
        
        Args:
            original_class_id: Original class ID from SMOD dataset
                - 0: person → person (1)
                - 1: rider → person (1)
                - 2: bicycle → vehicle (2)
                - 3: car → vehicle (2)
            
        Returns:
            New class ID: 1 for person, 2 for vehicle, or None to skip
        """
        if original_class_id == 0:
            # person → person (class 1)
            return 1
        elif original_class_id == 1:
            # rider → person (class 1)
            return 1
        elif original_class_id == 2:
            # bicycle → vehicle (class 2)
            return 2
        elif original_class_id == 3:
            # car → vehicle (class 2)
            return 2
        else:
            # Unknown class, skip
            return None
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of SMOD dataset
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # SMOD structure: images/ and labels/ folders
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.annotation_dir = os.path.join(root_dir, 'labels', split)
        
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.annotation_dir):
            raise ValueError(f"Annotation directory not found: {self.annotation_dir}")
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load annotations (YOLO format)
        base_name = os.path.splitext(self.image_files[idx])[0]
        ann_path = os.path.join(self.annotation_dir, base_name + '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            H, W = image.shape[:2]
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # YOLO format: class_id center_x center_y width height (normalized)
                    parts = line.split()
                    if len(parts) >= 5:
                        original_class_id = int(parts[0])
                        center_x = float(parts[1]) * W
                        center_y = float(parts[2]) * H
                        width = float(parts[3]) * W
                        height = float(parts[4]) * H
                        
                        # Convert to (x1, y1, x2, y2)
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        # Remap classes to 2-class system
                        new_class_id = self.remap_class(original_class_id)
                        if new_class_id is None:
                            # Skip this annotation if class mapping returns None
                            continue
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(new_class_id)  # Already includes background offset (1 or 2)
        
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


class DroneRGBTDataset(Dataset):
    """
    DroneRGBT Dataset Loader for RGB-Thermal paired images
    
    DroneRGBT contains paired RGB and Thermal images with synchronized annotations.
    Expected structure after preprocessing:
    dronergbt/
    ├── rgb/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── thermal/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── train/
        ├── val/
        └── test/
    
    Class Mapping (single-class system):
    - Only "person" class exists in the dataset
    - background (0), person (1)
    - Original class 0 (person) -> person (1)
    """
    
    # NOTE: Actual DroneRGBT dataset only contains "person" annotations (class 0)
    # Using single-class detection: person only
    CLASSES = [
        'background', 'person'
    ]
    
    @staticmethod
    def remap_class(original_class_id):
        """
        Remap original class ID to single-class system (person only)
        
        Args:
            original_class_id: Original class ID from dataset (0 = person)
            
        Returns:
            New class ID: 1 for person, or None to skip
        """
        # Dataset only has class 0 (person)
        if original_class_id == 0:
            # person -> person (class 1)
            return 1
        else:
            # Unknown class, skip
            return None
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of DroneRGBT dataset (after preprocessing)
            split: 'train', 'val', or 'test'
            transform: Optional transform to be applied on image pairs
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # DroneRGBT structure: rgb/ and thermal/ folders with matching names
        self.rgb_dir = os.path.join(root_dir, 'rgb', split)
        self.thermal_dir = os.path.join(root_dir, 'thermal', split)
        self.annotation_dir = os.path.join(root_dir, 'annotations', split)
        
        if not os.path.exists(self.rgb_dir):
            raise ValueError(f"RGB directory not found: {self.rgb_dir}")
        if not os.path.exists(self.thermal_dir):
            raise ValueError(f"Thermal directory not found: {self.thermal_dir}")
        if not os.path.exists(self.annotation_dir):
            raise ValueError(f"Annotation directory not found: {self.annotation_dir}")
        
        # Get all RGB image files (thermal should have matching names after preprocessing)
        self.image_files = sorted([
            f for f in os.listdir(self.rgb_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.rgb_dir}. Please run preprocessing first!")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, self.image_files[idx])
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_image = np.array(rgb_image)
        
        # Load corresponding thermal image
        thermal_path = os.path.join(self.thermal_dir, self.image_files[idx])
        if not os.path.exists(thermal_path):
            # Try different extensions
            base_name = os.path.splitext(self.image_files[idx])[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                thermal_path = os.path.join(self.thermal_dir, base_name + ext)
                if os.path.exists(thermal_path):
                    break
        
        thermal_image = Image.open(thermal_path)
        # Convert thermal to RGB if needed (for compatibility)
        if thermal_image.mode != 'RGB':
            thermal_image = thermal_image.convert('RGB')
        thermal_image = np.array(thermal_image)
        
        # Load annotations (shared for both modalities)
        base_name = os.path.splitext(self.image_files[idx])[0]
        ann_path = os.path.join(self.annotation_dir, base_name + '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            H, W = rgb_image.shape[:2]
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # YOLO format: class_id center_x center_y width height (normalized)
                    parts = line.split()
                    if len(parts) >= 5:
                        original_class_id = int(parts[0])
                        center_x = float(parts[1]) * W
                        center_y = float(parts[2]) * H
                        width = float(parts[3]) * W
                        height = float(parts[4]) * H
                        
                        # Convert to (x1, y1, x2, y2)
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        # Remap class (person only)
                        new_class_id = self.remap_class(original_class_id)
                        if new_class_id is None:
                            # Skip this annotation if class mapping returns None
                            continue
                        labels.append(new_class_id)  # person = 1 (background = 0)
                        
                        boxes.append([x1, y1, x2, y2])
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Convert images to tensors
        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        thermal_image = torch.from_numpy(thermal_image).permute(2, 0, 1).float() / 255.0
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        if self.transform:
            import copy
            # Seed RNG so both images get identical random transforms (same flip, same resize)
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            rgb_image, target = self.transform(rgb_image, target)
            # Deep-copy target before 2nd transform to avoid double-scaling boxes
            target_copy = copy.deepcopy(target)
            torch.manual_seed(seed)
            thermal_image, _ = self.transform(thermal_image, {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.int64)})
        
        return (rgb_image, thermal_image), target
