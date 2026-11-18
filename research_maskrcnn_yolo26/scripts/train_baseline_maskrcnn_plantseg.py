"""
End-to-End Training Suite for Canonical Baseline Mask R-CNN (ResNet-50-FPN) on PlantSeg (115 Categories)
Implements official canonical PyTorch/Torchvision fine-tuning protocol:
- Pre-trained ResNet-50-FPN backbone (COCO weights)
- Canonical FastRCNNPredictor and MaskRCNNPredictor for 116 classes
- Canonical SGD optimizer (lr=0.005, momentum=0.9, weight_decay=0.0005) + CosineAnnealingLR
- Batch size 4, image size 640x640
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from train_plantseg_hybrid import PlantSegDataset, collate_fn


def create_canonical_maskrcnn(num_classes: int = 116) -> nn.Module:
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    # 1. Replace Box Predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Replace Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


def train_baseline_maskrcnn(
    data_root: str = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/data/plantseg",
    output_dir: str = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/runs/plantseg_baseline_maskrcnn",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 0.005,
    num_classes: int = 116
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    weights_path = out_path / "weights"
    weights_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Datasets & Loaders
    train_dataset = PlantSegDataset(
        img_dir=Path(data_root) / "images" / "train",
        lbl_dir=Path(data_root) / "labels" / "train",
        img_size=640
    )
    val_dataset = PlantSegDataset(
        img_dir=Path(data_root) / "images" / "val",
        lbl_dir=Path(data_root) / "labels" / "val",
        img_size=640
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"[INFO] PlantSeg Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")
    print(f"[INFO] Number of classes: {num_classes} (115 plant diseases + 1 background)")

    # Instantiate model
    print("[INFO] Instantiating Canonical Pre-trained Mask R-CNN (ResNet-50-FPN)...")
    model = create_canonical_maskrcnn(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = []

    print(f"\n{'='*70}")
    print(f" STARTING CANONICAL BASELINE MASK R-CNN TRAINING ({epochs} EPOCHS)")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        roi_cls_loss = 0.0
        roi_box_loss = 0.0
        roi_mask_loss = 0.0
        rpn_cls_loss = 0.0
        rpn_box_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader):
            valid_pairs = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt["boxes"]) > 0]
            if len(valid_pairs) == 0:
                continue

            images = [img.to(device) for img, _ in valid_pairs]
            targets = [{k: v.to(device) for k, v in tgt.items()} for _, tgt in valid_pairs]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            train_loss += losses.item()
            roi_cls_loss += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
            roi_box_loss += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
            roi_mask_loss += loss_dict.get("loss_mask", torch.tensor(0.0)).item()
            rpn_cls_loss += loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
            rpn_box_loss += loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
            num_batches += 1

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch}/{epochs}] Batch [{step+1}/{len(train_loader)}] "
                    f"Loss: {losses.item():.4f} (Cls: {loss_dict.get('loss_classifier', 0):.3f}, "
                    f"Mask: {loss_dict.get('loss_mask', 0):.3f}, Box: {loss_dict.get('loss_box_reg', 0):.3f}) "
                    f"Rate: {num_batches / elapsed:.1f} batch/s"
                )

        scheduler.step()
        epoch_train_loss = train_loss / max(num_batches, 1)

        # Validation Loss
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                valid_pairs = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt["boxes"]) > 0]
                if len(valid_pairs) == 0:
                    continue
                images = [img.to(device) for img, _ in valid_pairs]
                targets = [{k: v.to(device) for k, v in tgt.items()} for _, tgt in valid_pairs]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_batches += 1

        epoch_val_loss = val_loss / max(val_batches, 1)
        epoch_time = time.time() - t0

        print(f"\n---> Epoch {epoch} Complete in {epoch_time:.1f}s | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(epoch_val_loss, 4),
            "roi_cls_loss": round(roi_cls_loss / max(num_batches, 1), 4),
            "roi_mask_loss": round(roi_mask_loss / max(num_batches, 1), 4),
            "roi_box_loss": round(roi_box_loss / max(num_batches, 1), 4),
            "time_sec": round(epoch_time, 2)
        }
        history.append(record)

        # Save checkpoint
        latest_ckpt = weights_path / "baseline_maskrcnn_plantseg_latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": epoch_val_loss
        }, str(latest_ckpt))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_ckpt = weights_path / "baseline_maskrcnn_plantseg_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss
            }, str(best_ckpt))
            print(f"[BEST] New best baseline model saved to: {best_ckpt} (Val Loss: {best_val_loss:.4f})\n")

    history_file = out_path / "baseline_training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[SUCCESS] Baseline Mask R-CNN training finished! Log saved to {history_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline Mask R-CNN on PlantSeg")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (canonical SGD default)")
    args = parser.parse_args()

    train_baseline_maskrcnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
