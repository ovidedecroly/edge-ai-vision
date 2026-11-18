import os
import sys
import re
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, '/media/ovide/DATA/PIT-UNS/edge-ai-vision')
from worker.recognizers.lprnet_recognizer import LPRNet, CHARS, CHAR_DICT, BLANK_IDX


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndonesianPlateDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]], img_size=(94, 24), is_train=True):
        self.samples = samples
        self.img_w, self.img_h = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.is_train:
            # Data augmentations: subtle brightness & contrast adjustment
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-10, 10)
            img_bgr = np.clip(alpha * img_bgr + beta, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        resized = cv2.resize(enhanced_bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype(np.float32) / 127.5 - 1.0
        return tensor.transpose(2, 0, 1)  # [3, H, W]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        tensor = self.preprocess(img)
        # Filter characters strictly into CHARS
        clean_label = re.sub(r'[^A-Z0-9]', '', label.upper())
        target = [CHAR_DICT[c] for c in clean_label if c in CHAR_DICT]
        return torch.from_numpy(tensor), torch.tensor(target, dtype=torch.long), clean_label


def collate_fn(batch):
    images, targets, raw_labels = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    concat_targets = torch.cat(targets, 0)
    return images, concat_targets, target_lengths, raw_labels


def decode_greedy_batch(logits: torch.Tensor) -> List[str]:
    """logits: [Batch, Class_Num, Time_Steps]"""
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy()  # [Batch, Time_Steps]

    results = []
    for seq in preds:
        char_list = []
        prev_idx = -1
        for idx in seq:
            if idx != prev_idx:
                if idx != BLANK_IDX:
                    char_list.append(CHARS[idx])
                prev_idx = idx
        results.append("".join(char_list))
    return results


def train():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Collect dataset samples
    kaggle_csv = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/raw_datasets/kaggle_linkgish_plates/plate_text_dataset/plate_text_dataset/label.csv"
    kaggle_dir = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/raw_datasets/kaggle_linkgish_plates/plate_text_dataset/plate_text_dataset/dataset"

    samples = []
    if os.path.exists(kaggle_csv):
        df = pd.read_csv(kaggle_csv)
        for _, row in df.iterrows():
            img_file = os.path.join(kaggle_dir, str(row["filename"]))
            if os.path.exists(img_file):
                samples.append((img_file, str(row["label"])))

    # Add local plate crops from storage/datasets/plates/labels.csv
    local_csv = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/datasets/plates/labels.csv"
    local_dir = "/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/datasets/plates"
    if os.path.exists(local_csv):
        df_loc = pd.read_csv(local_csv)
        for _, row in df_loc.iterrows():
            img_file = os.path.join(local_dir, str(row["image"]))
            if os.path.exists(img_file):
                samples.append((img_file, str(row["text"])))

    print(f"[INFO] Total dataset items: {len(samples)}")
    random.shuffle(samples)

    val_split = int(len(samples) * 0.15)
    train_samples = samples[val_split:]
    val_samples = samples[:val_split]

    train_ds = IndonesianPlateDataset(train_samples, is_train=True)
    val_ds = IndonesianPlateDataset(val_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 2. Build model & criterion
    model = LPRNet(class_num=len(CHARS)).to(device)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    save_dir = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/models/anpr")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_weight_path = save_dir / "lprnet_best.pth"

    best_acc = 0.0
    epochs = 25

    print(f"[INFO] Starting LPRNet training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            logits = model(images)  # [B, Class_Num, Time_Steps]
            time_steps = logits.size(2)

            # Permute to [Time_Steps, Batch, Class_Num] for CTCLoss
            log_probs = logits.permute(2, 0, 1).log_softmax(2)
            input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(1, n_batches)

        # Validation
        model.eval()
        total_val = 0
        exact_matches = 0

        with torch.no_grad():
            for images, _, _, raw_labels in val_loader:
                images = images.to(device)
                logits = model(images)
                preds = decode_greedy_batch(logits)

                for pred, gt in zip(preds, raw_labels):
                    clean_pred = re.sub(r'[^A-Z0-9]', '', pred)
                    clean_gt = re.sub(r'[^A-Z0-9]', '', gt)
                    if clean_pred == clean_gt:
                        exact_matches += 1
                    total_val += 1

        val_acc = (exact_matches / max(1, total_val)) * 100.0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - Val Exact Match: {val_acc:.2f}% ({exact_matches}/{total_val})")

        if val_acc >= best_acc or epoch == epochs:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "chars": CHARS
            }, str(best_weight_path))
            print(f"  -> Saved new best checkpoint to {best_weight_path} (Acc: {best_acc:.2f}%)")

    print(f"\n[DONE] LPRNet Training completed! Best Accuracy: {best_acc:.2f}%. Model saved at {best_weight_path}")


if __name__ == "__main__":
    train()
