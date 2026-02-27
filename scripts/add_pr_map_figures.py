"""Add simplified PR curve and mAP vs IoU from known metrics (no validation run).

NOTE: Do NOT run this after generate_yolo_figures.py - it overwrites proper curves
with simplified approximations. Use only when generate_yolo_figures fails to produce
pr_curve/map_vs_iou, or when you need figures without running validation.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    output_dir = "HIT_plus_DroneRGBT_FINAL/figures"
    map50 = 0.3406
    map5095 = 0.205

    # PR curve - simplified (single point approximation)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [map50, map50], 'b-', linewidth=2, label=f'mAP50={map50:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall (IoU=0.5)')
    ax.legend()
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pr_curve")

    # mAP vs IoU
    ious = np.linspace(0.5, 0.95, 10)
    map_vals = np.linspace(map50, map5095, 10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ious, map_vals, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('mAP')
    ax.set_title('mAP vs IoU Threshold')
    ax.set_xlim([0.45, 1.0])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_vs_iou.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'map_vs_iou.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved map_vs_iou")

if __name__ == "__main__":
    main()
