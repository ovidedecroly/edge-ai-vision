import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO

from config import PREPARED_ROOT, CHECKPOINT_ROOT, FIGURE_ROOT, OUTPUT_ROOT, resolve_dataset_dir, get_device
from models import create_model
from utils import torch_checkpoint_state, ensure_dirs
from dataset_v7 import PlantSegCOCODataset

def overlay(img, masks, boxes, labels, scores=None, alpha=.45):
    out=img.copy()
    rng=np.random.default_rng(42)
    for i,m in enumerate(masks):
        color=rng.integers(40,230,3,dtype=np.uint8)
        mm=m.astype(bool)
        out[mm]=(out[mm]*(1-alpha)+color*alpha).astype(np.uint8)
        x1,y1,x2,y2=map(int,boxes[i])
        cv2.rectangle(out,(x1,y1),(x2,y2),tuple(map(int,color.tolist())),2)
    return out

def find_yolo_best():
    c = list((OUTPUT_ROOT / "yolo_runs").rglob("weights/best.pt"))
    return max(c, key=lambda p: p.stat().st_mtime) if c else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=6)
    args = ap.parse_args()
    ensure_dirs(FIGURE_ROOT)
    device = get_device()
    ds = PlantSegCOCODataset("test")
    coco = COCO(str(PREPARED_ROOT / "coco" / "annotation_test.clean.json"))

    models = []
    for t in [1, 2, 3]:
        ckpt = CHECKPOINT_ROOT / f"tier{t}" / "best.pt"
        if ckpt.exists():
            m = create_model(t).to(device)
            m.load_state_dict(torch_checkpoint_state(ckpt, device), strict=True)
            m.eval()
            models.append((t, m))

    yolo_ckpt = find_yolo_best()
    yolo_model = YOLO(str(yolo_ckpt)) if yolo_ckpt else None

    cols = 2 + len(models) + (1 if yolo_model else 0)
    idxs = np.linspace(0, len(ds)-1, args.num_images, dtype=int)
    fig, axes = plt.subplots(args.num_images, cols, figsize=(3.6 * cols, 3.7 * args.num_images))
    if args.num_images == 1:
        axes = np.expand_dims(axes, 0)

    tier_names = {
        1: "Tier 1 (Canonical Mask R-CNN)",
        2: "Tier 2 (YOLO26-FPN Mask R-CNN)",
        3: "Tier 3 (Fast Mask R-CNN ProtoNet)",
    }

    for r, idx in enumerate(idxs):
        img, t = ds[idx]
        rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_masks = t["masks"].numpy().astype(bool)
        gt_boxes = t["boxes"].numpy()

        c_idx = 0
        axes[r, c_idx].imshow(rgb)
        axes[r, c_idx].set_title("Input")
        c_idx += 1

        axes[r, c_idx].imshow(overlay(rgb, gt_masks, gt_boxes, t["labels"].numpy()))
        axes[r, c_idx].set_title("Ground Truth")
        c_idx += 1

        for tier, m in models:
            with torch.no_grad():
                p = m([img.to(device)])[0]
            keep = (p["scores"] >= 0.25).cpu().numpy()
            masks = (p["masks"][keep, 0].cpu().numpy() > 0.5)
            boxes = p["boxes"][keep].cpu().numpy()
            labels = p["labels"][keep].cpu().numpy()
            axes[r, c_idx].imshow(overlay(rgb, masks, boxes, labels))
            axes[r, c_idx].set_title(tier_names.get(tier, f"Tier {tier}"))
            c_idx += 1

        if yolo_model is not None:
            r_yolo = yolo_model.predict(rgb, imgsz=640, conf=0.25, verbose=False)[0]
            if r_yolo.boxes is not None and len(r_yolo.boxes) > 0 and r_yolo.masks is not None:
                y_boxes = r_yolo.boxes.xyxy.cpu().numpy()
                y_labels = r_yolo.boxes.cls.cpu().numpy().astype(int)
                y_masks = r_yolo.masks.data.cpu().numpy()
                resized_masks = np.stack([
                    cv2.resize(m.astype(np.float32), (rgb.shape[1], rgb.shape[0])) > 0.5
                    for m in y_masks
                ])
                axes[r, c_idx].imshow(overlay(rgb, resized_masks, y_boxes, y_labels))
            else:
                axes[r, c_idx].imshow(rgb)
            axes[r, c_idx].set_title("Tier 4 (YOLO26-seg)")
            c_idx += 1

        for c in range(cols):
            axes[r, c].axis("off")

    plt.tight_layout()
    out = FIGURE_ROOT / "qualitative_real_predictions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved", out)

if __name__=="__main__":
    main()
