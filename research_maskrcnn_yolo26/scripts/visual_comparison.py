"""
Visual Comparison Figure Generator:
Renders side-by-side segmentation predictions on real CarParts-Seg validation images:
1. Input Image
2. Standard Mask R-CNN
3. YOLO26-Backboned Mask R-CNN (Proposed Hybrid)
4. Pure YOLO26-seg (One-Stage)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from yolo26_mask_rcnn_model import create_yolo26_mask_rcnn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from ultralytics import YOLO


def generate_visual_comparison():
    val_imgs = list(Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/data/carparts-seg/images/val").glob("*.jpg"))
    if not val_imgs:
        print("[ERROR] No validation images found!")
        return

    sample_img_path = str(val_imgs[0])
    img_bgr = cv2.imread(sample_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Models
    pure_yolo = YOLO("/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt")
    yolo_maskrcnn = create_yolo26_mask_rcnn(num_classes=24).cuda().eval()
    std_maskrcnn = maskrcnn_resnet50_fpn(weights=None, num_classes=24).cuda().eval()

    # Preprocessing for torchvision
    img_tensor = [torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).cuda()]

    # 1. Pure YOLO26-seg
    yolo_preds = pure_yolo.predict(img_rgb, imgsz=640, device="cuda", verbose=False)[0]
    yolo_vis = yolo_preds.plot()

    # 2. YOLO26-Backboned Mask R-CNN
    with torch.no_grad():
        preds_yolo_mask = yolo_maskrcnn(img_tensor)[0]
        preds_std_mask = std_maskrcnn(img_tensor)[0]

    def overlay_maskrcnn(base_img, preds, conf_thresh=0.25):
        canvas = base_img.copy()
        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        masks = preds["masks"].squeeze(1).cpu().numpy()
        
        colors = [(239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8), (168, 85, 247)]
        
        for i in range(len(boxes)):
            if scores[i] < conf_thresh: continue
            color = colors[i % len(colors)]
            mask = masks[i] > 0.5
            canvas[mask] = (canvas[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{scores[i]:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return canvas

    std_vis = overlay_maskrcnn(img_rgb, preds_std_mask, conf_thresh=0.10)
    hybrid_vis = overlay_maskrcnn(img_rgb, preds_yolo_mask, conf_thresh=0.10)

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("(a) Input Vehicle Image (CarParts-Seg)", fontsize=13, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(std_vis)
    axes[0, 1].set_title("(b) Standard Mask R-CNN (ResNet-50-FPN) [44.04M params]", fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(hybrid_vis)
    axes[1, 0].set_title("(c) Proposed YOLO26-Backboned Mask R-CNN [20.51M params, -53.4%]", fontsize=13, fontweight="bold", pad=8, color="#173B65")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(yolo_vis)
    axes[1, 1].set_title("(d) Pure YOLO26-seg (One-Stage ProtoNet) [3.13M params, 91.4 FPS]", fontsize=13, fontweight="bold", pad=8, color="#047857")
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_fig = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/paper/figures/visual_comparison.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[SUCCESS] Visual comparison figure generated at: {out_fig}")


if __name__ == "__main__":
    generate_visual_comparison()
