"""
Generates convergence_comparison.png for the PlantSeg benchmark across:
1. Standard Mask R-CNN (ResNet-50-FPN) [Canonical Baseline]
2. YOLO26-Backboned Mask R-CNN [Hybrid Stage 1]
3. Fast YOLO26-Mask R-CNN [Hybrid Stage 2 - ProtoNet + Light-Head]
4. Pure YOLO26-seg [One-Stage SOTA]

Figure 2 showcases:
(a) 50-Epoch Training Loss Dynamics across all 4 architectural tiers
(b) 50-Epoch Validation Loss Trajectory & Generalization Gap
(c) Speed-Accuracy Pareto Frontier: GPU Throughput (FPS) vs. Best Validation Loss (Bubble size = Parameters)
Ensures ZERO overlap with Table 1 (which reports static numerical hardware benchmark metrics).
"""

import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Plot styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0

RUNS_DIR = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/runs/production_plantseg")
PAPER_DIR = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/paper")
BRAIN_DIRS = [
    Path("/home/ovide/.gemini/antigravity-ide/brain/78fe0816-5e93-4e25-9c1c-61067f488b7a"),
    Path("/home/ovide/.gemini/antigravity-ide/brain/e1faf51d-a0dc-4666-8a80-ec7f2008b927")
]

def load_data():
    # Tier 1
    with open(RUNS_DIR / "tier1_standard_mask_r_cnn_(resnet_50_fpn)" / "history.json") as f:
        h1 = json.load(f)
    # Tier 2
    with open(RUNS_DIR / "tier2_yolo26_backboned_mask_r_cnn" / "history.json") as f:
        h2 = json.load(f)
    # Tier 3
    with open(RUNS_DIR / "tier3_fast_light_head_yolo26_mask_r_cnn" / "history.json") as f:
        h3 = json.load(f)
    # Tier 4
    with open(RUNS_DIR / "tier4_pure_yolo26_seg" / "results.csv") as f:
        r4 = list(csv.DictReader(f))

    epochs1 = [x["epoch"] for x in h1]
    train1 = [x["train_loss"] for x in h1]
    val1 = [x["val_loss"] for x in h1]

    epochs2 = [x["epoch"] for x in h2]
    train2 = [x["train_loss"] for x in h2]
    val2 = [x["val_loss"] for x in h2]

    epochs3 = [x["epoch"] for x in h3]
    train3 = [x["train_loss"] for x in h3]
    val3 = [x["val_loss"] for x in h3]

    epochs4 = [int(x["epoch"]) for x in r4]
    train4 = [float(x["train/box_loss"]) + float(x["train/seg_loss"]) + float(x["train/cls_loss"]) for x in r4]
    val4 = [float(x["val/box_loss"]) + float(x["val/seg_loss"]) + float(x["val/cls_loss"]) for x in r4]

    return {
        "t1": (epochs1, train1, val1),
        "t2": (epochs2, train2, val2),
        "t3": (epochs3, train3, val3),
        "t4": (epochs4, train4, val4)
    }

def main():
    data = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    plt.subplots_adjust(wspace=0.25, left=0.06, right=0.96, top=0.88, bottom=0.14)

    colors = {
        "t1": "#ef4444", # Red
        "t2": "#f59e0b", # Amber
        "t3": "#10b981", # Emerald
        "t4": "#3b82f6"  # Blue
    }

    labels = {
        "t1": "Tier 1: Standard Mask R-CNN (ResNet-50)",
        "t2": "Tier 2: YOLO26-Backbone Mask R-CNN",
        "t3": "Tier 3: Fast YOLO26-Mask R-CNN (Ours)",
        "t4": "Tier 4: Pure YOLO26-seg (One-Stage)"
    }

    # -------------------------------------------------------------
    # (a) Training Loss Trajectory (Epoch 1 to 50)
    # -------------------------------------------------------------
    ax1 = axes[0]
    for k in ["t1", "t2", "t3", "t4"]:
        epochs, train, _ = data[k]
        lw = 2.4 if k == "t3" else 1.8
        alpha = 1.0 if k == "t3" else 0.85
        ax1.plot(epochs, train, color=colors[k], label=labels[k], linewidth=lw, alpha=alpha)

    ax1.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Training Loss", fontsize=11, fontweight="bold")
    ax1.set_title("(a) 50-Epoch Training Loss Dynamics", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim(1, 50)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # (b) Validation Loss Trajectory & Generalization
    # -------------------------------------------------------------
    ax2 = axes[1]
    for k in ["t1", "t2", "t3", "t4"]:
        epochs, _, val = data[k]
        lw = 2.4 if k == "t3" else 1.8
        alpha = 1.0 if k == "t3" else 0.85
        ax2.plot(epochs, val, color=colors[k], label=labels[k], linewidth=lw, alpha=alpha)

    # Annotate best validation points for Tier 2 and Tier 3
    min_val2 = min(data["t2"][2])
    min_ep2 = data["t2"][0][data["t2"][2].index(min_val2)]
    ax2.scatter(min_ep2, min_val2, color=colors["t2"], s=60, zorder=5)
    ax2.annotate(f"Tier 2 Best: {min_val2:.4f}", (min_ep2, min_val2),
                 textcoords="offset points", xytext=(10, 8),
                 fontsize=8.5, fontweight="bold", color="#b45309",
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1))

    min_val3 = min(data["t3"][2])
    min_ep3 = data["t3"][0][data["t3"][2].index(min_val3)]
    ax2.scatter(min_ep3, min_val3, color=colors["t3"], s=60, zorder=5)
    ax2.annotate(f"Tier 3 Best: {min_val3:.4f}", (min_ep3, min_val3),
                 textcoords="offset points", xytext=(10, -15),
                 fontsize=8.5, fontweight="bold", color="#047857",
                 arrowprops=dict(arrowstyle="->", color="#047857", lw=1))

    ax2.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Loss", fontsize=11, fontweight="bold")
    ax2.set_title("(b) 50-Epoch Validation Loss & Convergence", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(1, 50)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------
    # (c) Pareto Frontier: GPU Throughput (FPS) vs Best Val Loss
    # -------------------------------------------------------------
    ax3 = axes[2]
    fps = [26.5, 56.3, 125.4, 84.9]
    best_val_losses = [1.1758, 1.0318, 1.0685, 3.1122]
    params = [44.54, 21.00, 10.38, 3.13]
    tier_names = [
        "1. Standard Mask R-CNN\n(44.5M, 26.5 FPS)",
        "2. YOLO26-Backbone\n(21.0M, 56.3 FPS)",
        "3. Fast YOLO26-Mask (Ours)\n(10.4M, 125.4 FPS)",
        "4. Pure YOLO26-seg\n(3.1M, 84.9 FPS)"
    ]
    point_colors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"]

    # Bubble scatter plot where size indicates parameters
    bubble_sizes = [p * 28 + 120 for p in params]
    scatter = ax3.scatter(fps, best_val_losses, s=bubble_sizes, c=point_colors, alpha=0.75, edgecolors="#1e293b", linewidth=1.5, zorder=4)

    # Draw Pareto boundary line between best trade-offs (Tier 2, Tier 3)
    ax3.plot([56.3, 125.4], [1.0318, 1.0685], linestyle=":", color="#059669", linewidth=2.0, zorder=3, label="Optimal Pareto Frontier")

    # Annotate labels
    offsets = [(-45, 14), (-40, 16), (-45, -28), (-55, -24)]
    for i in range(4):
        ax3.annotate(tier_names[i], (fps[i], best_val_losses[i]),
                     textcoords="offset points", xytext=offsets[i],
                     fontsize=8.5, fontweight="bold", color="#1e293b",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.9))

    ax3.set_xlabel("Inference Throughput (GPU FPS, RTX 3070 Ti) [Higher is better]", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Best Validation Loss [Lower is better]", fontsize=10.5, fontweight="bold")
    ax3.set_title("(c) Pareto Frontier: Throughput vs. Loss (Bubble size = Params)", fontsize=12, fontweight="bold", pad=10)
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Save to all target paths
    out_files = [
        PAPER_DIR / "convergence_comparison.png",
        BRAIN_DIRS[0] / "convergence_comparison.png",
        BRAIN_DIRS[1] / "convergence_comparison.png"
    ]
    for out in out_files:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=280, bbox_inches="tight")
        print(f"[SUCCESS] Unified convergence plot saved to: {out}")

    plt.close()

if __name__ == "__main__":
    main()
