"""
PlantSeg v7 dataset for Mask R-CNN-family models (Tiers 1-3), V2.

Changes w.r.t. V1 (all documented in runs/plantseg_v7_v2/geometry_audit.json):
  * NO forced 640x640 stretch.  The ORIGINAL-resolution image is returned (uint8, CHW);
    torchvision's GeneralizedRCNNTransform performs the aspect-preserving resize
        s = min(Smin/min(H,W), Smax/max(H,W)),  (H',W') = (sH, sW)
    and rescales boxes + masks consistently.  Nothing is resized manually.
  * Images are always read with cv2.imread default (EXIF orientation applied).  The audit showed that
    polygons live in this oriented frame for every EXIF case.
  * Masks are rasterised from polygons at the oriented (H,W) via pycocotools (same rasteriser as annToMask),
    NOT cv2.fillPoly and NOT with any cv2.resize repair.  GT json = coco_geomfix (only h/w of 8 EXIF-swapped
    images corrected).
  * TRAIN view excludes images whose polygons overshoot the image frame by >2% (geometry_flags.json).
    val / test are never filtered.
  * Instance-safe augmentation (train only): hflip p=0.5, mild colour jitter, [scale jitter is applied by the
    RCNN transform through multi-scale min_size so boxes/masks are updated by torchvision itself].
  * Memory guard (train only): images above MAX_TRAIN_PIXELS (4 Mpix, ~3% of train) are uniformly
    area-downscaled together with their masks (nearest) before leaving the loader.  Val/test images are never
    resized and eval targets carry no masks (GT comes from COCO json).
  * Test split is blocked unless config.ALLOW_TEST_EVAL is True.
"""
import json
import random
from collections import Counter
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from pycocotools import mask as mask_utils

from config import resolve_dataset_dir, V2_ROOT, V2_AUDITS, ALLOW_TEST_EVAL

MAX_TRAIN_PIXELS = 4_000_000
GEOMFIX_DIR = V2_ROOT / "coco_geomfix"


def geomfix_json_path(split):
    return GEOMFIX_DIR / f"annotation_{split}.clean.json"


def load_geomfix_coco(split):
    return json.loads(geomfix_json_path(split).read_text())


def load_flags():
    return json.loads((V2_AUDITS / "geometry_flags.json").read_text())


def guard_split(split):
    if split == "test" and not ALLOW_TEST_EVAL:
        raise PermissionError(
            "Test split is locked in V2 development (config.ALLOW_TEST_EVAL=False). "
            "Unlock only after the user approves a frozen architecture.")


def polygons_to_mask(segmentation, h, w):
    """Union of all polygons of one annotation, rasterised by pycocotools at (h,w). No resizing."""
    polys = [p for p in segmentation if len(p) >= 6 and len(p) % 2 == 0]
    if not polys:
        return None
    rles = mask_utils.frPyObjects(polys, h, w)
    return mask_utils.decode(mask_utils.merge(rles))


def to_float_image(img):
    """uint8 CHW -> float CHW in [0,1] (device preserved)."""
    return img.float().div_(255.0) if img.dtype == torch.uint8 else img


class PlantSegCOCODataset(Dataset):
    def __init__(self, split: str, train: Optional[bool] = None, exclude_flagged: Optional[bool] = None,
                 load_masks: Optional[bool] = None, subset: Optional[list] = None):
        guard_split(split)
        self.split = split
        self.train = (split == "train") if train is None else train
        self.exclude_flagged = (split == "train") if exclude_flagged is None else exclude_flagged
        self.load_masks = self.train if load_masks is None else load_masks
        self.augment = self.train
        self.ds = resolve_dataset_dir()
        self.img_dir = self.ds / "images" / split
        coco = load_geomfix_coco(split)
        excl = set(load_flags()[split]["flagged_inconsistent"]) if self.exclude_flagged else set()
        self.images = [im for im in coco["images"] if im["file_name"] not in excl]
        self.n_excluded = len(coco["images"]) - len(self.images)
        keep = {im["id"] for im in self.images}
        self.by_image = {}
        for ann in coco["annotations"]:
            if ann["image_id"] in keep:
                self.by_image.setdefault(ann["image_id"], []).append(ann)
        if subset is not None:
            self.images = [self.images[i] for i in subset]
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)

    def __len__(self):
        return len(self.images)

    # ------------------------------------------------------------------ repeat-factor sampling (Part 17)
    def class_instance_counts(self, n_classes=115):
        c = Counter()
        for im in self.images:
            for a in self.by_image.get(im["id"], []):
                c[int(a["category_id"])] += 1
        return np.array([c.get(i, 0) for i in range(n_classes)], dtype=np.int64)

    def repeat_factors(self, t: float):
        """r_c = max(1, sqrt(t/f_c)),  f_c = n_c / sum_j n_j (instance frequency), r_I = max_{c in I} r_c."""
        n = self.class_instance_counts().astype(np.float64)
        f_c = n / max(1.0, n.sum())
        r_c = np.where(n > 0, np.maximum(1.0, np.sqrt(t / np.maximum(f_c, 1e-12))), 1.0)
        return np.array([max([r_c[int(a["category_id"])] for a in self.by_image.get(im["id"], [])] or [1.0])
                         for im in self.images])

    def repeat_indices(self, t: float, seed: int):
        rng = np.random.RandomState(seed)
        rf = self.repeat_factors(t)
        idx = []
        for i, r in enumerate(rf):
            n = int(np.floor(r)) + int(rng.rand() < (r - np.floor(r)))
            idx.extend([i] * n)
        return idx

    # ------------------------------------------------------------------ item
    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = info["id"]
        p = self.img_dir / info["file_name"]
        img = cv2.imread(str(p))                    # EXIF orientation applied (audited as the polygon frame)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        assert (H, W) == (info["height"], info["width"]), (
            f"{info['file_name']}: image {(H, W)} != geomfix json {(info['height'], info['width'])}")

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        if self.load_masks:
            f = 1.0
            if self.train and H * W > MAX_TRAIN_PIXELS:
                f = float(np.sqrt(MAX_TRAIN_PIXELS / (H * W)))
            for ann in self.by_image.get(image_id, []):
                seg = ann.get("segmentation", [])
                if not isinstance(seg, list) or len(seg) == 0:
                    continue
                m = polygons_to_mask(seg, H, W)
                if m is None or m.sum() == 0:
                    continue
                if f != 1.0:
                    m = cv2.resize(m, (max(1, int(round(W * f))), max(1, int(round(H * f)))),
                                   interpolation=cv2.INTER_NEAREST)
                    if m.sum() == 0:
                        continue
                ys, xs = np.where(m)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1  # tight box of rasterised mask
                boxes.append([x1, y1, x2, y2])
                labels.append(int(ann["category_id"]) + 1)  # torchvision: background = 0
                masks.append(m)
                areas.append(float(m.sum()))
                iscrowd.append(int(ann.get("iscrowd", 0)))
            if f != 1.0:
                img = cv2.resize(img, (max(1, int(round(W * f))), max(1, int(round(H * f)))), interpolation=cv2.INTER_AREA)

        h2, w2 = img.shape[:2]
        if boxes:
            boxes = torch.tensor(np.array(boxes), dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.from_numpy(np.stack(masks)).to(torch.uint8)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h2, w2), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()  # uint8 CHW

        if self.augment:                                 # instance-safe augmentation, targets updated in sync
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
                masks = masks.flip(-1)
                if len(boxes):
                    boxes = torch.stack([w2 - boxes[:, 2], boxes[:, 1], w2 - boxes[:, 0], boxes[:, 3]], 1)
            if random.random() < 0.8:
                tensor = self.color_jitter(tensor)

        target = {
            "boxes": boxes, "labels": labels, "masks": masks,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas, "iscrowd": iscrowd,
            "orig_size": torch.tensor([H, W], dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
