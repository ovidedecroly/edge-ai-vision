import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from worker.recognizers.ocr_recognizer import BaseRecognizer, normalize_indonesian_plate

# Standard LPRNet Character set: Digits, Uppercase Latin, and Blank/Padding token '-'
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '-'
]
CHAR_DICT = {char: i for i, char in enumerate(CHARS)}
BLANK_IDX = len(CHARS) - 1  # '-' is CTC blank


class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LPRNet(nn.Module):
    """
    Lightweight License Plate Recognition Network (LPRNet).
    Input image dimension: [Batch, 3, 24, 94]
    Output: Character sequence logits [Batch, Class_Num, Time_Steps]
    """
    def __init__(self, class_num: int = len(CHARS), dropout_rate: float = 0.5):
        super().__init__()
        self.class_num = class_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.block1 = SmallBasicBlock(64, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))

        self.block2 = SmallBasicBlock(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.block3 = SmallBasicBlock(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.drop = nn.Dropout(dropout_rate)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 4), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(256, class_num, kernel_size=(1, 1))

        # Multi-scale global context feature fusion
        self.container = nn.Conv2d(64 + 64 + class_num, class_num, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale 1
        x = self.conv1(x)
        x = self.pool1(x)
        f1 = self.relu1(self.bn1(self.block1(x)))

        # Scale 2
        x = self.pool2(f1)
        x = self.relu2(self.bn2(self.block2(x)))
        f2 = self.relu3(self.bn3(self.block3(x)))

        # Scale 3
        x = self.pool3(f2)
        x = self.drop(x)
        x = self.conv2(x)
        f3 = self.conv3(x)

        # Upsample multi-scale feature maps to match f3 time-steps
        target_h, target_w = f3.shape[2], f3.shape[3]
        f1_down = F.adaptive_avg_pool2d(f1, (target_h, target_w))
        f2_down = F.adaptive_avg_pool2d(f2, (target_h, target_w))

        # Concatenate multi-scale representations
        cat_features = torch.cat([f1_down, f2_down, f3], dim=1)
        logits = self.container(cat_features)

        # Average pool across height to get [Batch, Class_Num, Time_Steps]
        logits = logits.mean(dim=2)
        return logits


class LPRNetRecognizer(BaseRecognizer):
    """
    Indonesian License Plate Recognizer using LPRNet.
    Preprocesses cropped license plate:
    - CLAHE contrast enhancement
    - Aspect ratio resize to (94, 24)
    - Forward pass through LPRNet
    - CTC Greedy Decoding with Korlantas Polri post-disambiguation
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.model_id = "lprnet-indonesia-v1"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LPRNet(class_num=len(CHARS)).to(self.device)
        self.model.eval()

        if weights_path and os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=self.device)
                if "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"], strict=False)
                elif "state_dict" in state:
                    self.model.load_state_dict(state["state_dict"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[WARN] Failed to load LPRNet weights from {weights_path}: {e}")

    def preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Resize to 94x24, apply CLAHE, normalize to [-1, 1]."""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        resized = cv2.resize(enhanced_bgr, (94, 24), interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype(np.float32) / 127.5 - 1.0
        tensor = tensor.transpose(2, 0, 1)  # [3, 24, 94]
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def decode_greedy(self, logits: torch.Tensor) -> Tuple[str, float, List[float]]:
        """
        CTC Greedy Decoding on [1, Class_Num, Time_Steps]:
        - Argmax over Class_Num at each time step
        - Collapse consecutive identical characters
        - Filter out blank tokens ('-')
        """
        probs = F.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        preds = preds[0].cpu().numpy()
        max_probs = max_probs[0].cpu().numpy()

        char_list = []
        conf_list = []
        prev_idx = -1

        for idx, p in zip(preds, max_probs):
            if idx != prev_idx:
                if idx != BLANK_IDX:
                    char_list.append(CHARS[idx])
                    conf_list.append(float(p))
                prev_idx = idx

        raw_str = "".join(char_list)
        formatted_plate = self._format_indonesian_plate(raw_str)
        overall_conf = float(np.mean(conf_list)) if conf_list else 0.0
        return formatted_plate, overall_conf, conf_list

    def _format_indonesian_plate(self, raw_str: str) -> str:
        """
        Applies Korlantas Polri structural rules:
        Area Code (1-2 letters) + Space + 1-4 Digits + Space + 1-3 Sub-district letters
        e.g. AD2914JG -> AD 2914 JG
        """
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_str.upper())
        if not cleaned:
            return ""

        match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})?$', cleaned)
        if match:
            area, num, suffix = match.groups()
            return f"{area} {num}" + (f" {suffix}" if suffix else "")

        # Fallback segmentation
        m_area = re.match(r'^[A-Z]{1,2}', cleaned)
        if m_area:
            area = m_area.group(0)
            rest = cleaned[len(area):]
            m_nums = re.match(r'^\d+', rest)
            if m_nums:
                num = m_nums.group(0)
                suffix = rest[len(num):]
                return f"{area} {num}" + (f" {suffix}" if suffix else "")

        return cleaned

    def recognize(self, plate_crop_bgr: np.ndarray) -> Dict[str, Any]:
        start = time.perf_counter()
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return {
                "text": "",
                "confidence": 0.0,
                "character_confidences": [],
                "processing_time_ms": 0.0,
                "model_id": self.model_id
            }

        inp = self.preprocess(plate_crop_bgr)
        with torch.no_grad():
            logits = self.model(inp)
            text, conf, char_confs = self.decode_greedy(logits)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "text": text,
            "confidence": round(conf, 4),
            "character_confidences": [round(c, 4) for c in char_confs],
            "processing_time_ms": round(elapsed_ms, 2),
            "model_id": self.model_id
        }
