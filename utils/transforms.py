"""
Data augmentation and transformation utilities
"""

import torch
import torchvision.transforms.functional as F
from torchvision import transforms


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL image to tensor"""
    def __call__(self, image, target):
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    """Randomly flip image horizontally"""
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = F.hflip(image)
            if 'boxes' in target and target['boxes'].numel() > 0:
                boxes = target['boxes'].clone()
                # Flip x coordinates: x_center -> width - x_center
                width = image.shape[2]
                boxes[:, 0] = width - boxes[:, 0]
                target['boxes'] = boxes
        return image, target


class Resize:
    """Resize image while maintaining aspect ratio"""
    def __init__(self, max_size=1536, multi_scale=False):
        self.max_size = max_size
        self.multi_scale = multi_scale
        if multi_scale:
            # Multi-scale training: randomly sample from [1024, 1280, 1536]
            self.scales = [1024, 1280, 1536]
    
    def __call__(self, image, target):
        h, w = image.shape[1], image.shape[2]
        
        # Multi-scale training: randomly sample max_size
        if self.multi_scale:
            max_size = self.scales[torch.randint(0, len(self.scales), (1,)).item()]
        else:
            max_size = self.max_size
        
        # Calculate scale
        scale = min(max_size / h, max_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        image = F.resize(image, (new_h, new_w))
        
        # Scale boxes
        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes = target['boxes'].clone()
            boxes[:, 0] *= scale  # x_center
            boxes[:, 1] *= scale  # y_center
            boxes[:, 2] *= scale  # width
            boxes[:, 3] *= scale  # height
            target['boxes'] = boxes
        
        return image, target


class Pad:
    """Pad image to fixed size"""
    def __init__(self, size=(1536, 1536), fill=0):
        self.size = size
        self.fill = fill
    
    def __call__(self, image, target):
        h, w = image.shape[1], image.shape[2]
        max_h, max_w = self.size
        
        # Calculate padding
        pad_h = max(0, max_h - h)
        pad_w = max(0, max_w - w)
        
        if pad_h > 0 or pad_w > 0:
            # Pad: (left, right, top, bottom)
            image = F.pad(image, (0, pad_w, 0, pad_h), fill=self.fill)
        
        return image, target


def get_train_transform(max_size=1536, multi_scale=False):
    """Get training transforms with data augmentation"""
    # Use multi-scale if enabled (randomly sample from [1024, 1280, 1536])
    # Pad to maximum size to ensure consistent batch shapes
    max_pad_size = max_size if not multi_scale else 1536
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(prob=0.5),
        Resize(max_size=max_size, multi_scale=multi_scale),
        Pad(size=(max_pad_size, max_pad_size), fill=0)
    ])


def get_val_transform(max_size=1536):
    """Get validation transforms (no augmentation)"""
    return Compose([
        ToTensor(),
        Resize(max_size=max_size),
        Pad(size=(max_size, max_size), fill=0)
    ])

