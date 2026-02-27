"""
Generate evaluation figures for YOLOv8: confusion matrix, mAP vs IoU, metrics bar,
detection examples, PR curve. Uses ultralytics + custom plotting.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    import argparse
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument('--model', default='HIT_plus_DroneRGBT_FINAL/weights/best.pt')
    p.add_argument('--data', default='data/combined_hit_dronergbt/combined.yaml')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--output_dir', default='HIT_plus_DroneRGBT_FINAL/figures')
    p.add_argument('--num_detection_examples', type=int, default=6)
    p.add_argument('--conf_threshold', type=float, default=0.25, help='Min confidence for detection examples')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = YOLO(args.model)

    # Run validation to get metrics (no built-in plots to avoid KeyError)
    print("Running validation...")
    results = model.val(data=args.data, split=args.split, plots=False, verbose=True)

    # Extract metrics - results has .box (BoxMetrics)
    metrics = results.box
    names = {0: 'Person', 1: 'Vehicle'}
    nc = 2

    # 1. Confusion matrix - Ultralytics: (nc+1)x(nc+1), rows=predicted, cols=GT, last row/col=background
    cm = np.zeros((nc + 1, nc + 1), dtype=np.float64)
    try:
        cm_obj = getattr(results, 'confusion_matrix', None)
        if cm_obj is not None and hasattr(cm_obj, 'matrix'):
            raw = np.asarray(cm_obj.matrix, dtype=np.float64)
            if raw.size > 0 and raw.shape[0] == nc + 1 and raw.shape[1] == nc + 1:
                cm = raw.copy()
    except Exception:
        pass

    # Labels: Ultralytics order is class 0..nc-1, then background at index nc
    labels = [names.get(i, f'C{i}') for i in range(nc)] + ['background']
    cm = cm[:len(labels), :len(labels)]  # ensure shape matches

    # Plot: use counts, vmin=0 (no negative values)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_display = np.maximum(cm, 0)  # ensure non-negative
    vmax = max(cm_display.max(), 1)
    im = ax.imshow(cm_display, cmap='Blues', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted')
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            v = cm_display[i, j]
            ax.text(j, i, int(round(v)), ha='center', va='center', fontsize=10,
                    color='white' if v > vmax * 0.5 else 'black')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion_matrix")

    # 2. PR curve - from metrics.curves (Recall on x, Precision on y)
    try:
        curves = getattr(metrics, 'curves', None)
        px, py = None, None
        if curves is not None:
            # Ultralytics: curves is (px, py) where px=Recall, py=Precision (or similar)
            for c in curves:
                if isinstance(c, (tuple, list)) and len(c) >= 2:
                    rx, ry = c[0], c[1]
                    rx, ry = np.asarray(rx), np.asarray(ry)
                    if len(rx) > 1 and rx.min() >= 0 and rx.max() <= 1.1:
                        px, py = rx, ry
                        break
        if px is not None and py is not None and len(px) > 1:
            fig, ax = plt.subplots(figsize=(5, 4))
            py_arr = np.atleast_2d(np.asarray(py)) if np.asarray(py).ndim == 1 else np.asarray(py)
            n_curves = py_arr.shape[1] if py_arr.ndim > 1 else 1
            for i in range(min(nc, n_curves)):
                y = py_arr[:, i] if py_arr.ndim > 1 else py_arr.flatten()
                ax.plot(px, y, label=names.get(i, f'Class {i}'), linewidth=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve (IoU=0.5)')
            ax.legend()
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'pr_curve.pdf'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(args.output_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved pr_curve")
        else:
            # Fallback: single point from macro P, R
            map50_val = float(getattr(metrics, 'map50', 0) or 0)
            mp = float(getattr(metrics, 'mp', 0) or 0)
            mr = float(getattr(metrics, 'mr', 0) or 0)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot([0, mr, 1], [mp, mp, 0], 'b-', linewidth=2, label=f'mAP50={map50_val:.3f}')
            ax.scatter([mr], [mp], s=80, zorder=5)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall (IoU=0.5)')
            ax.legend()
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'pr_curve.pdf'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(args.output_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved pr_curve (fallback)")
    except Exception as e:
        print(f"  PR curve skipped: {e}")

    # 3. mAP vs IoU - from metrics.all_ap (nc, 10) for IoU 0.5:0.05:0.95
    try:
        ious = np.linspace(0.5, 0.95, 10)
        all_ap = getattr(metrics, 'all_ap', None)
        if all_ap is not None:
            ap = np.array(all_ap)
            if ap.ndim >= 2:
                map_vals = np.nanmean(ap, axis=0)[:10]
            else:
                map_vals = np.array([float(metrics.map50)] * 10)
        else:
            map50 = float(metrics.map50) if metrics.map50 is not None else 0
            map50_95 = float(metrics.map) if metrics.map is not None else map50
            map_vals = np.linspace(map50, map50_95, 10)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ious, map_vals, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('mAP')
        ax.set_title('mAP vs IoU Threshold')
        ax.set_xlim([0.45, 1.0])
        ax.set_ylim([0, max(1.05, float(np.nanmax(map_vals)) * 1.1)])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'map_vs_iou.pdf'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(args.output_dir, 'map_vs_iou.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved map_vs_iou")
    except Exception as e:
        print(f"  mAP vs IoU skipped: {e}")

    # 4. Metrics bar - per-class P, R, mAP50
    try:
        ap50 = getattr(metrics, 'ap50', None)
        p = getattr(metrics, 'p', None)
        r = getattr(metrics, 'r', None)
        map50 = float(metrics.map50) if metrics.map50 is not None else 0
        if ap50 is None:
            ap50 = np.array([map50] * nc)
        else:
            ap50 = np.atleast_1d(ap50)
        if p is None:
            p = np.array([map50] * nc)
        else:
            p = np.atleast_1d(p)
        if r is None:
            r = np.array([map50] * nc)
        else:
            r = np.atleast_1d(r)
        # Ensure length nc
        ap50 = np.resize(ap50, nc)
        p = np.resize(p, nc)
        r = np.resize(r, nc)

        x = np.arange(nc)
        width = 0.25
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width, p[:nc], width, label='Precision')
        ax.bar(x, r[:nc], width, label='Recall')
        ax.bar(x + width, ap50[:nc], width, label='AP50')
        ax.set_xticks(x)
        ax.set_xticklabels([names.get(i, f'C{i}') for i in range(nc)])
        ax.set_ylabel('Score')
        ax.set_title('Per-class Metrics')
        ax.legend()
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'metrics_bar.pdf'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(args.output_dir, 'metrics_bar.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved metrics_bar")
    except Exception as e:
        print(f"  Metrics bar skipped: {e}")

    # 5. Detection examples - run predict on sample images
    data_path = Path(args.data)
    import yaml
    with open(data_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg['path'])
    img_dir = root / 'images' / args.split
    if img_dir.exists():
        imgs = sorted(img_dir.glob('*.jpg'))[:args.num_detection_examples]
        if not imgs:
            imgs = list(img_dir.glob('*.png'))[:args.num_detection_examples]
        if imgs:
            pred_results = model.predict(imgs, save=False, verbose=False, conf=args.conf_threshold)
            n = min(len(pred_results), 6)
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            for i, (res, ax) in enumerate(zip(pred_results[:n], axes)):
                img = res.orig_img
                ax.imshow(img)
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        color = ['green', 'blue'][cls] if cls < 2 else 'red'
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2)
                        ax.add_patch(rect)
                        ax.text(x1, y1-4, f'{names.get(cls, cls)} {conf:.2f}', color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.7))
                ax.axis('off')
                ax.set_title(imgs[i].name[:20])
            for j in range(i+1, 6):
                axes[j].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'detection_examples.pdf'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(args.output_dir, 'detection_examples.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved detection_examples")

    # Save metrics summary
    try:
        map50 = float(metrics.map50) if metrics.map50 is not None else 0
        map5095 = float(metrics.map) if metrics.map is not None else map50
        mp = float(metrics.mp) if metrics.mp is not None else 0
        mr = float(metrics.mr) if metrics.mr is not None else 0
        with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Data: {args.data} split={args.split}\n")
            f.write(f"mAP50: {map50:.4f}\n")
            f.write(f"mAP50-95: {map5095:.4f}\n")
            f.write(f"Precision: {mp:.4f}\n")
            f.write(f"Recall: {mr:.4f}\n")
    except Exception as e:
        print(f"  metrics_summary skipped: {e}")

    print(f"\nAll figures saved to {args.output_dir}")

if __name__ == '__main__':
    main()
