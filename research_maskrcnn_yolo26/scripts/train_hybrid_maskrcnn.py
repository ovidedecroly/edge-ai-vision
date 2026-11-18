"""
End-to-End Training Suite for YOLO26-Backboned Mask R-CNN on CarParts-Seg
Hybrid Two-Stage Instance Segmentation Architecture Training Pipeline.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from yolo26_mask_rcnn_model import create_yolo26_mask_rcnn


class CarPartsDataset(Dataset):
    """
    Dataset loader for CarParts-Seg YOLO formatted annotations converted to torchvision Mask R-CNN targets.
    """
    def __init__(self, img_dir: Path, lbl_dir: Path, img_size: int = 640):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.img_files = sorted([p for p in self.img_dir.glob("*.jpg")] + [p for p in self.img_dir.glob("*.png")])

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.img_files[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to fixed size
        if h != self.img_size or w != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
            scale_x = self.img_size / w
            scale_y = self.img_size / h
        else:
            scale_x, scale_y = 1.0, 1.0

        boxes = []
        labels = []
        masks = []

        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # 1-indexed for Mask R-CNN (0 is background)
                    cls_id = int(parts[0]) + 1
                    coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                    coords[:, 0] *= self.img_size
                    coords[:, 1] *= self.img_size

                    x1 = np.min(coords[:, 0])
                    y1 = np.min(coords[:, 1])
                    x2 = np.max(coords[:, 0])
                    y2 = np.max(coords[:, 1])

                    if (x2 - x1) < 2 or (y2 - y1) < 2:
                        continue

                    mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)
                    masks.append(mask)

        # Handle empty annotations safely
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            masks = np.zeros((0, self.img_size, self.img_size), dtype=np.uint8)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.array(masks, dtype=np.uint8)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8)
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_hybrid_maskrcnn(
    data_root: str = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/data/carparts-seg",
    yolo_weights: str = "/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt",
    output_dir: str = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/runs/train_hybrid",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_classes: int = 24
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    weights_path = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/weights")
    weights_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Datasets & Loaders
    train_dataset = CarPartsDataset(
        img_dir=Path(data_root) / "images" / "train",
        lbl_dir=Path(data_root) / "labels" / "train",
        img_size=640
    )
    val_dataset = CarPartsDataset(
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

    print(f"[INFO] Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # Instantiate model
    print("[INFO] Instantiating YOLO26-Backboned Mask R-CNN...")
    model = create_yolo26_mask_rcnn(
        num_classes=num_classes,
        yolo_weights=yolo_weights,
        fpn_channels=256,
        pretrained_backbone=True
    )
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = []

    print(f"\n{'='*70}")
    print(f" STARTING HYBRID YOLO26-MASK R-CNN TRAINING ({epochs} EPOCHS)")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        rpn_cls_loss = 0.0
        rpn_box_loss = 0.0
        roi_cls_loss = 0.0
        roi_box_loss = 0.0
        roi_mask_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader):
            # Skip empty target batches
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

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
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
        # In torchvision, setting model.train() while withholding gradient computes val loss
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

        # Record history
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

        # Save latest checkpoint
        latest_ckpt = weights_path / "yolo26_maskrcnn_latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": epoch_val_loss
        }, str(latest_ckpt))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_ckpt = weights_path / "yolo26_maskrcnn_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss
            }, str(best_ckpt))
            print(f"[BEST] New best model saved to: {best_ckpt} (Val Loss: {best_val_loss:.4f})\n")

    # Save training history JSON
    import json
    history_file = out_path / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[SUCCESS] Training finished! Log saved to {history_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26-Mask R-CNN on CarParts-Seg")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    train_hybrid_maskrcnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
