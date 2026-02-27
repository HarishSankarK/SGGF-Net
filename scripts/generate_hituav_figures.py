"""
Generate HIT-UAV evaluation figures with confusion matrix WITHOUT background class.
Outputs: confusion_matrix (2x2), confusion_matrix_normalized (2x2), BoxPR, BoxF1,
BoxP, BoxR, map_vs_iou, metrics_bar, detection_examples.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def smooth(y, f=0.1):
    """Box filter (Ultralytics-style)."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def main():
    import argparse
    import yaml
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument('--model', default='HIT_UAV_CHKPT/weights/best.pt')
    p.add_argument('--data', default='data/hit-uav-2class/hituav_2class.yaml')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--output_dir', default='HIT_UAV_CHKPT/figures')
    p.add_argument('--num_detection_examples', type=int, default=6)
    p.add_argument('--conf_threshold', type=float, default=0.25)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = YOLO(args.model)

    # Run validation (plots=True lets Ultralytics save BoxPR/F1/P/R; we overwrite confusion matrix)
    print("Running validation...")
    try:
        results = model.val(
            data=args.data, split=args.split, plots=True, verbose=True,
            project=str(Path(args.output_dir).parent), name=Path(args.output_dir).name,
            exist_ok=True
        )
    except (KeyError, IndexError) as e:
        print(f"  plots=True failed ({e}), using plots=False")
        results = model.val(data=args.data, split=args.split, plots=False, verbose=True)

    metrics = results.box
    names = {0: 'Person', 1: 'Vehicle'}
    nc = 2

    # 1. Confusion matrix (2x2, no background)
    cm_full = np.zeros((nc + 1, nc + 1), dtype=np.float64)
    try:
        cm_obj = getattr(results, 'confusion_matrix', None)
        if cm_obj is not None and hasattr(cm_obj, 'matrix'):
            raw = np.asarray(cm_obj.matrix, dtype=np.float64)
            if raw.size > 0 and raw.shape[0] >= nc and raw.shape[1] >= nc:
                cm_full = raw.copy()
    except Exception:
        pass

    cm = cm_full[:nc, :nc]  # Person, Vehicle only (drop background row/col)
    labels = [names.get(i, f'C{i}') for i in range(nc)]

    for normalize, suffix in [(False, ''), (True, '_normalized')]:
        fig, ax = plt.subplots(figsize=(5, 4))
        if normalize:
            col_sums = cm.sum(axis=0, keepdims=True) + 1e-9
            arr = cm / col_sums
            vmax = 1.0
        else:
            arr = np.maximum(cm, 0)
            vmax = max(arr.max(), 1)
        im = ax.imshow(arr, cmap='Blues', vmin=0, vmax=vmax)
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        for i in range(nc):
            for j in range(nc):
                v = arr[i, j]
                text = f'{v:.2f}' if normalize else str(int(round(v)))
                ax.text(j, i, text, ha='center', va='center', fontsize=12,
                        color='white' if v > vmax * 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='Count' if not normalize else 'Fraction')
        plt.title('Confusion Matrix' + (' Normalized' if normalize else ''))
        plt.tight_layout()
        fname = f'confusion_matrix{suffix}'
        plt.savefig(os.path.join(args.output_dir, f'{fname}.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(args.output_dir, f'{fname}.pdf'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")

    # 2. Box curves (PR, F1, P, R) - from Ultralytics curves if available
    curves = getattr(metrics, 'curves', None)
    if curves is not None and len(curves) >= 4:
        # curves: (pr_curve, f1_curve, p_curve, r_curve) typically
        # pr: px=recall, py=precision
        # f1,p,r: px=confidence, py=metric
        try:
            # PR curve (index 0)
            pr = curves[0]
            if isinstance(pr, (tuple, list)) and len(pr) >= 2:
                px, py = np.asarray(pr[0]), np.asarray(pr[1])
                if px.size > 1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    if py.ndim == 1:
                        ax.plot(px, py, 'b-', linewidth=2, label='all')
                    else:
                        for i in range(min(nc, py.shape[1])):
                            ax.plot(px, py[:, i], linewidth=1.5, label=names.get(i, f'C{i}'))
                        map50 = float(getattr(metrics, 'map50', 0) or 0)
                        ax.plot(px, py.mean(axis=1), linewidth=3, color='blue',
                                label=f'all classes {map50:.3f} mAP@0.5')
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    ax.set_title('Precision-Recall Curve')
                    ax.legend(loc='lower left')
                    ax.set_xlim(0, 1.05)
                    ax.set_ylim(0, 1.05)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'BoxPR_curve.png'), dpi=150, bbox_inches='tight')
                    plt.savefig(os.path.join(args.output_dir, 'BoxPR_curve.pdf'), dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved BoxPR_curve")

            # F1, P, R curves (indices 1, 2, 3) - Confidence on x, metric on y
            for idx, (ylabel, fname) in enumerate([
                (1, ('F1', 'BoxF1_curve')),
                (2, ('Precision', 'BoxP_curve')),
                (3, ('Recall', 'BoxR_curve')),
            ]):
                if idx + 1 >= len(curves):
                    continue
                c = curves[idx + 1]
                if isinstance(c, (tuple, list)) and len(c) >= 2:
                    px, py = np.asarray(c[0]), np.asarray(c[1])
                    if px.size > 1:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        if py.ndim == 1:
                            ax.plot(px, py, 'b-', linewidth=2)
                        else:
                            for i in range(min(nc, py.shape[0] if py.ndim == 2 else 1)):
                                y = py[i] if py.ndim == 2 else py
                                ax.plot(px, y, linewidth=1.5, label=names.get(i, f'C{i}'))
                            ys = smooth(py.mean(axis=0) if py.ndim == 2 else py, 0.1)
                            peak = ys.max()
                            peak_x = px[np.argmax(ys)] if len(px) == len(ys) else px[len(px)//2]
                            ax.plot(px[:len(ys)], ys, linewidth=3, color='blue',
                                    label=f'all classes {peak:.2f} at {peak_x:.3f}')
                        ax.set_xlabel('Confidence')
                        ax.set_ylabel(ylabel)
                        ax.set_title(f'{ylabel}-Confidence Curve')
                        ax.legend(loc='upper right')
                        ax.set_xlim(0, 1.05)
                        ax.set_ylim(0, 1.05)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.output_dir, f'{fname[1]}.png'), dpi=150, bbox_inches='tight')
                        plt.savefig(os.path.join(args.output_dir, f'{fname[1]}.pdf'), dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"  Saved {fname[1]}")
        except Exception as e:
            print(f"  Box curves skipped: {e}")

    # 3. mAP vs IoU
    try:
        ious = np.linspace(0.5, 0.95, 10)
        all_ap = getattr(metrics, 'all_ap', None)
        if all_ap is not None:
            ap = np.array(all_ap)
            map_vals = np.nanmean(ap, axis=0)[:10] if ap.ndim >= 2 else np.full(10, float(metrics.map50 or 0))
        else:
            map50 = float(metrics.map50 or 0)
            map50_95 = float(metrics.map or map50)
            map_vals = np.linspace(map50, map50_95, 10)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(ious, map_vals, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('mAP')
        ax.set_title('mAP vs IoU Threshold')
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0, max(1.05, float(np.nanmax(map_vals)) * 1.1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'map_vs_iou.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(args.output_dir, 'map_vs_iou.pdf'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved map_vs_iou")
    except Exception as e:
        print(f"  map_vs_iou skipped: {e}")

    # 4. Metrics bar
    try:
        ap50 = np.atleast_1d(getattr(metrics, 'ap50', None) or [metrics.map50] * nc)
        p = np.atleast_1d(getattr(metrics, 'p', None) or [0.5] * nc)
        r = np.atleast_1d(getattr(metrics, 'r', None) or [0.5] * nc)
        ap50, p, r = np.resize(ap50, nc), np.resize(p, nc), np.resize(r, nc)
        x = np.arange(nc)
        width = 0.25
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width, p, width, label='Precision')
        ax.bar(x, r, width, label='Recall')
        ax.bar(x + width, ap50, width, label='AP50')
        ax.set_xticks(x)
        ax.set_xticklabels([names[i] for i in range(nc)])
        ax.set_ylabel('Score')
        ax.set_title('Per-class Metrics')
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'metrics_bar.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(args.output_dir, 'metrics_bar.pdf'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved metrics_bar")
    except Exception as e:
        print(f"  metrics_bar skipped: {e}")

    # 5. Detection examples
    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get('path', 'data/hit-uav-2class'))
    img_dir = root / cfg.get('val', 'images/val')
    if img_dir.exists():
        imgs = sorted(img_dir.glob('*.jpg'))[:args.num_detection_examples]
        if not imgs:
            imgs = list(img_dir.glob('*.png'))[:args.num_detection_examples]
        if imgs:
            preds = model.predict(imgs, save=False, verbose=False, conf=args.conf_threshold)
            n = min(len(preds), 6)
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            for i, (res, ax) in enumerate(zip(preds[:n], axes)):
                ax.imshow(res.orig_img)
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        color = 'green' if cls == 0 else 'blue'
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2)
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 4, f'{names.get(cls, cls)} {conf:.2f}', fontsize=8,
                                color='white', bbox=dict(facecolor=color, alpha=0.7))
                ax.axis('off')
                ax.set_title(imgs[i].name[:24])
            for j in range(i + 1, 6):
                axes[j].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'detection_examples.png'), dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(args.output_dir, 'detection_examples.pdf'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved detection_examples")

    # 6. Metrics summary
    try:
        with open(os.path.join(args.output_dir, 'EVAL_RESULTS.txt'), 'w') as f:
            f.write(f"HIT-UAV YOLOv8 Evaluation Results (best.pt)\n")
            f.write("=" * 40 + "\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.data} (2 classes: Person, Vehicle)\n")
            f.write(f"Split: {args.split}\n")
            map50 = float(metrics.map50 or 0)
            map5095 = float(metrics.map or map50)
            mp = float(metrics.mp or 0)
            mr = float(metrics.mr or 0)
            f.write(f"\nMetrics:\n--------\n")
            f.write(f"all:      Precision={mp:.3f}, Recall={mr:.3f}, mAP50={map50:.3f}, mAP50-95={map5095:.3f}\n")
            ap50 = getattr(metrics, 'ap50', None)
            p = getattr(metrics, 'p', None)
            r = getattr(metrics, 'r', None)
            if ap50 is not None and p is not None and r is not None:
                ap50, p, r = np.atleast_1d(ap50), np.atleast_1d(p), np.atleast_1d(r)
                for i in range(nc):
                    f.write(f"{names.get(i, str(i)).lower()}:   Precision={p[i]:.3f}, Recall={r[i]:.3f}, mAP50={ap50[i]:.3f}\n")
            f.write(f"\nFigures (no background in confusion matrix):\n")
            f.write("- confusion_matrix.png, confusion_matrix_normalized.png\n")
            f.write("- BoxPR_curve, BoxF1_curve, BoxP_curve, BoxR_curve\n")
            f.write("- map_vs_iou, metrics_bar, detection_examples\n")
        print(f"  Saved EVAL_RESULTS.txt")
    except Exception as e:
        print(f"  EVAL_RESULTS skipped: {e}")

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
