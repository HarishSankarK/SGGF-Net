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

    # Run validation to get metrics — plots=True ensures confusion_matrix is populated
    print("Running validation...")
    results = model.val(data=args.data, split=args.split, plots=True, verbose=True)

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
            if raw.size > 0:
                if raw.shape[0] == nc + 1 and raw.shape[1] == nc + 1:
                    cm = raw.copy()
                elif raw.shape[0] == nc and raw.shape[1] == nc:
                    # Pad with empty background row/col
                    cm[:nc, :nc] = raw
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

    # 6. F1-Confidence curve
    try:
        curves_data = getattr(metrics, 'curves_results', None)
        f1_found = False
        if curves_data is not None:
            # curves_results: list of (x, y, xlabel, ylabel) tuples
            for item in curves_data:
                if len(item) >= 4 and 'F1' in str(item[3]):
                    px_f1, py_f1 = np.asarray(item[0]), np.asarray(item[1])
                    fig, ax = plt.subplots(figsize=(5, 4))
                    py_f1_2d = np.atleast_2d(py_f1) if py_f1.ndim == 1 else py_f1
                    for i in range(min(nc, py_f1_2d.shape[0] if py_f1_2d.ndim > 1 else 1)):
                        y = py_f1_2d[i] if py_f1_2d.ndim > 1 else py_f1_2d.flatten()
                        ax.plot(px_f1, y, label=names.get(i, f'Class {i}'), linewidth=2)
                    ax.set_xlabel('Confidence Threshold')
                    ax.set_ylabel('F1 Score')
                    ax.set_title('F1-Confidence Curve')
                    ax.legend()
                    ax.set_xlim([0, 1.0])
                    ax.set_ylim([0, 1.05])
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'f1_confidence.pdf'), dpi=150, bbox_inches='tight')
                    plt.savefig(os.path.join(args.output_dir, 'f1_confidence.png'), dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved f1_confidence")
                    f1_found = True
                    break
        if not f1_found:
            # Fallback: compute F1 from P and R arrays if available
            p_arr = np.atleast_1d(getattr(metrics, 'p', None) or [])
            r_arr = np.atleast_1d(getattr(metrics, 'r', None) or [])
            if len(p_arr) > 0 and len(r_arr) > 0:
                f1_vals = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-9)
                fig, ax = plt.subplots(figsize=(5, 4))
                class_labels = [names.get(i, f'Class {i}') for i in range(min(nc, len(f1_vals)))]
                ax.bar(class_labels, f1_vals[:nc], color=['steelblue', 'coral'][:nc])
                ax.set_ylabel('F1 Score')
                ax.set_title('F1 Score per Class')
                ax.set_ylim([0, 1.05])
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'f1_confidence.pdf'), dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(args.output_dir, 'f1_confidence.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved f1_confidence (bar fallback)")
    except Exception as e:
        print(f"  F1-Confidence curve skipped: {e}")

    # 7. Training curves (loss + mAP over epochs) from results.csv
    try:
        import argparse as _ap
        import csv
        # Try to find results.csv from the training run directory
        # User can pass --train_dir; default: look relative to model path
        model_path = Path(args.model)
        candidate_dirs = [
            model_path.parent.parent,           # weights/../  = run dir
            model_path.parent.parent.parent,     # one level up
        ]
        results_csv = None
        for d in candidate_dirs:
            csv_path = d / 'results.csv'
            if csv_path.exists():
                results_csv = csv_path
                break

        if results_csv is not None:
            epochs_list, box_loss, cls_loss, dfl_loss, map50_list, map_list = [], [], [], [], [], []
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                # Strip whitespace from keys
                for row in reader:
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    try:
                        epochs_list.append(int(float(row.get('epoch', 0))))
                        # Losses
                        box_loss.append(float(row.get('train/box_loss', row.get('box_loss', 0))))
                        cls_loss.append(float(row.get('train/cls_loss', row.get('cls_loss', 0))))
                        dfl_loss.append(float(row.get('train/dfl_loss', row.get('dfl_loss', 0))))
                        # Metrics
                        map50_list.append(float(row.get('metrics/mAP50(B)', row.get('mAP50', 0))))
                        map_list.append(float(row.get('metrics/mAP50-95(B)', row.get('mAP50-95', 0))))
                    except (ValueError, KeyError):
                        continue

            if epochs_list:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Loss curves
                axes[0].plot(epochs_list, box_loss, label='Box Loss', linewidth=2)
                axes[0].plot(epochs_list, cls_loss, label='Cls Loss', linewidth=2)
                axes[0].plot(epochs_list, dfl_loss, label='DFL Loss', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss Curves')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # mAP curves
                axes[1].plot(epochs_list, map50_list, label='mAP@50', linewidth=2, color='green')
                axes[1].plot(epochs_list, map_list, label='mAP@50-95', linewidth=2, color='orange')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('mAP')
                axes[1].set_title('Validation mAP over Epochs')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                axes[1].set_ylim([0, 1.05])

                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'training_curves.pdf'), dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved training_curves")
        else:
            print(f"  training_curves skipped: results.csv not found near {args.model}")
    except Exception as e:
        print(f"  Training curves skipped: {e}")

    # 8. Per-class AP at multiple IoU thresholds
    try:
        all_ap = getattr(metrics, 'all_ap', None)
        if all_ap is not None:
            ap = np.array(all_ap)  # (nc, 10)
            if ap.ndim == 2 and ap.shape[1] >= 3:
                ious_sel = [0, 2, 4, 6, 9]  # IoU 0.50, 0.60, 0.70, 0.80, 0.95
                iou_labels = ['0.50', '0.60', '0.70', '0.80', '0.95']
                x = np.arange(len(ious_sel))
                width = 0.35
                fig, ax = plt.subplots(figsize=(7, 4))
                colors = ['steelblue', 'coral']
                for i in range(min(nc, ap.shape[0])):
                    vals = ap[i, ious_sel]
                    offset = (i - nc / 2 + 0.5) * width
                    ax.bar(x + offset, vals, width, label=names.get(i, f'Class {i}'), color=colors[i % len(colors)])
                ax.set_xticks(x)
                ax.set_xticklabels([f'IoU={t}' for t in iou_labels])
                ax.set_ylabel('AP')
                ax.set_title('Per-class AP at Multiple IoU Thresholds')
                ax.legend()
                ax.set_ylim([0, 1.05])
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'ap_per_iou.pdf'), dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(args.output_dir, 'ap_per_iou.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved ap_per_iou")
    except Exception as e:
        print(f"  AP per IoU skipped: {e}")

    # 9. Confidence score distribution (TP vs FP)
    try:
        data_path_obj = Path(args.data)
        import yaml as _yaml
        with open(data_path_obj) as f:
            cfg2 = _yaml.safe_load(f)
        root2 = Path(cfg2['path'])
        img_dir2 = root2 / 'images' / args.split
        all_imgs = sorted(list(img_dir2.glob('*.jpg')) + list(img_dir2.glob('*.png')))
        sample_imgs = all_imgs[:50]  # sample 50 images for speed

        if sample_imgs:
            pred_results2 = model.predict(sample_imgs, save=False, verbose=False, conf=0.01)
            tp_scores, fp_scores = [], []
            for res in pred_results2:
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                # Load corresponding label file
                lbl_path = Path(str(res.path).replace('/images/', '/labels/'))
                lbl_path = lbl_path.with_suffix('.txt')
                if not lbl_path.exists():
                    continue
                gt_boxes = []
                with open(lbl_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, cx, cy, w, h = map(float, parts)
                            H, W = res.orig_img.shape[:2]
                            x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
                            x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
                            gt_boxes.append([x1, y1, x2, y2])
                if not gt_boxes:
                    continue
                gt_arr = np.array(gt_boxes)
                matched = set()
                for box in res.boxes:
                    conf = float(box.conf[0])
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                    ious_box = []
                    for gi, gb in enumerate(gt_arr):
                        ix1 = max(px1, gb[0]); iy1 = max(py1, gb[1])
                        ix2 = min(px2, gb[2]); iy2 = min(py2, gb[3])
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        area_p = (px2-px1)*(py2-py1); area_g = (gb[2]-gb[0])*(gb[3]-gb[1])
                        union = area_p + area_g - inter
                        ious_box.append(inter / (union + 1e-7))
                    best_gi = int(np.argmax(ious_box))
                    if ious_box[best_gi] >= 0.5 and best_gi not in matched:
                        tp_scores.append(conf)
                        matched.add(best_gi)
                    else:
                        fp_scores.append(conf)

            if tp_scores or fp_scores:
                fig, ax = plt.subplots(figsize=(6, 4))
                bins = np.linspace(0, 1, 25)
                if tp_scores:
                    ax.hist(tp_scores, bins=bins, alpha=0.7, label=f'TP (n={len(tp_scores)})', color='green')
                if fp_scores:
                    ax.hist(fp_scores, bins=bins, alpha=0.7, label=f'FP (n={len(fp_scores)})', color='red')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Count')
                ax.set_title('Confidence Score Distribution (TP vs FP)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'score_distribution.pdf'), dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(args.output_dir, 'score_distribution.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved score_distribution")
    except Exception as e:
        print(f"  Score distribution skipped: {e}")

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
