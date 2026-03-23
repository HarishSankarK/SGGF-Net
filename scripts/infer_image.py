"""
Run Fusion-YOLOv11 inference on a single image and save the result with detections.
Usage:
  python scripts/infer_image.py --image path/to/image.jpg --checkpoint checkpoints/best.pth [--output out.jpg]
"""

import os
import sys
import argparse
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import FusionYOLOv11
from utils.transforms import get_val_transform

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


CLASS_NAMES = {1: 'Person', 2: 'Vehicle'}
COLORS = {1: (0, 255, 0), 2: (0, 0, 255)}  # Green, Blue


def is_ultralytics_checkpoint_path(checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception:
        return checkpoint_path.endswith('.pt')
    return isinstance(ckpt, dict) and 'train_args' in ckpt and 'ema' in ckpt


def ultralytics_predict(checkpoint_path, image_np, conf_threshold, nms_threshold, device):
    if YOLO is None:
        raise ImportError("Ultralytics is not installed, so .pt checkpoints cannot be loaded here.")
    device_arg = 'cpu' if device.type == 'cpu' else str(device)
    load_path = checkpoint_path
    temp_path = None
    if not checkpoint_path.endswith('.pt'):
        fd, temp_path = tempfile.mkstemp(suffix='.pt')
        os.close(fd)
        shutil.copy2(checkpoint_path, temp_path)
        load_path = temp_path
    model = YOLO(load_path, task='detect')
    result = model.predict(
        source=image_np,
        conf=conf_threshold,
        iou=nms_threshold,
        device=device_arg,
        verbose=False
    )[0]
    if temp_path is not None:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    boxes = result.boxes
    if boxes is None or boxes.xyxy.numel() == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.long),
        }
    return {
        'boxes': boxes.xyxy.detach().cpu(),
        'scores': boxes.conf.detach().cpu(),
        'labels': boxes.cls.detach().cpu().to(torch.long) + 1,
    }


def main():
    parser = argparse.ArgumentParser(description='Infer on single image')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path (.pth)')
    parser.add_argument('--output', type=str, default=None, help='Output image path (default: input_detected.jpg)')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold (higher=stricter, fewer FPs). Try 0.5-0.6 for cleaner output.')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                        help='NMS IoU threshold (lower=more aggressive duplicate removal). Try 0.3-0.45.')
    parser.add_argument('--max_detections', type=int, default=50,
                        help='Max detections per image (reduces clutter)')
    parser.add_argument('--min_box_area', type=float, default=0,
                        help='Min box area (pixels²) to keep. 0=disabled. Try 400-900 to filter tiny FPs.')
    parser.add_argument('--max_box_area_ratio', type=float, default=0.95,
                        help='Max box area as fraction of image. Boxes larger are dropped. 1.0=disabled.')
    parser.add_argument('--max_size', type=int, default=640)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if not os.path.exists(args.image):
        print(f'Error: Image not found: {args.image}')
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f'Error: Checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.image)
        output_path = f'{base}_detected{ext}'

    # Load image
    img_pil = Image.open(args.image).convert('RGB')
    img_np = np.array(img_pil)
    orig_h, orig_w = img_np.shape[:2]

    # Transform (same as validation)
    transform = get_val_transform(max_size=args.max_size)
    target = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.long)}
    img_tensor, _ = transform(img_pil, target)
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    scale = min(args.max_size / orig_h, args.max_size / orig_w)

    is_ultra = is_ultralytics_checkpoint_path(args.checkpoint)

    if is_ultra:
        preds = ultralytics_predict(
            args.checkpoint, img_np, args.conf_threshold, args.nms_threshold, device
        )
    else:
        # Load checkpoint first to get backbone and num_classes (so model architecture matches)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        backbone = ckpt.get('backbone', 'resnet50') if isinstance(ckpt, dict) else 'resnet50'
        model_num_classes = ckpt.get('num_classes') if isinstance(ckpt, dict) else None
        if model_num_classes is None:
            model_num_classes = 2 if args.num_classes >= 3 else args.num_classes
        model = FusionYOLOv11(num_classes=model_num_classes, pretrained=False, fusion_type='concat_attention', backbone=backbone)
        model.load_state_dict(ckpt.get('ema_state_dict', ckpt['model_state_dict']))
        model = model.to(device).eval()

        # Predict (SMOD/RGB: use same image for both streams)
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            preds = model.predict(inp, inp, conf_threshold=args.conf_threshold,
                                  nms_threshold=args.nms_threshold)[0]

    # Apply max_detections (keep top by score)
    if len(preds['boxes']) > args.max_detections:
        top_idx = torch.argsort(preds['scores'], descending=True)[:args.max_detections]
        preds = {k: v[top_idx] for k, v in preds.items()}

    # Optional box size filtering (removes oversized/small junk without retraining)
    if args.min_box_area > 0 or args.max_box_area_ratio < 1.0:
        areas = (preds['boxes'][:, 2] - preds['boxes'][:, 0]) * (preds['boxes'][:, 3] - preds['boxes'][:, 1])
        img_area = H * W
        keep = (areas >= args.min_box_area) & (areas <= img_area * args.max_box_area_ratio)
        preds = {k: v[keep] for k, v in preds.items()}

    # Draw on the original image. Fusion-model predictions are in resized/padded coords,
    # so map them back by undoing the resize scale. Padding is only on right/bottom.
    preds_to_draw = preds
    if not is_ultra and len(preds['boxes']) > 0:
        boxes = preds['boxes'].clone()
        boxes = boxes / scale
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, orig_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, orig_h)
        preds_to_draw = {
            'boxes': boxes,
            'scores': preds['scores'],
            'labels': preds['labels'],
        }

    img_out = img_np.copy()
    # Use matplotlib for drawing (consistent with eval figures)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_out)
    for i in range(len(preds_to_draw['boxes'])):
        x1, y1, x2, y2 = preds_to_draw['boxes'][i].cpu().tolist()
        score = preds_to_draw['scores'][i].item()
        lab = preds_to_draw['labels'][i].item()
        color = ['green', 'blue'][lab - 1] if lab in (1, 2) else 'red'
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f'{CLASS_NAMES.get(lab, lab)} {score:.2f}', color='white',
                fontsize=10, bbox=dict(facecolor=color, alpha=0.7))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')
    print(f'Detections: {len(preds["boxes"])} (Person: {(preds["labels"]==1).sum().item()}, Vehicle: {(preds["labels"]==2).sum().item()})')


if __name__ == '__main__':
    main()
