"""
Evaluation helpers (V2).  Predictions of the torchvision-style models are already mapped back to the ORIGINAL
(EXIF-oriented) image grid by GeneralizedRCNNTransform.postprocess; nothing is rescaled here.
GT comes from the geometry-corrected COCO json; shape mismatches RAISE instead of being resized away.
"""
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from config import NUM_DISEASE_CLASSES
from dataset_v7 import geomfix_json_path, guard_split


def build_coco(split):
    guard_split(split)
    return COCO(str(geomfix_json_path(split)))


def gt_semantic_map(coco: COCO, image_id, h, w):
    """background=0, disease class c -> c+1.  Later annotations overwrite earlier ones (same rule as V1)."""
    sem = np.zeros((h, w), dtype=np.int32)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
        m = coco.annToMask(ann)
        if m.shape != (h, w):
            raise ValueError(f"image {image_id}: GT mask {m.shape} != image {(h, w)} (geometry not corrected)")
        sem[m.astype(bool)] = int(ann["category_id"]) + 1
    return sem


def predictions_to_coco(pred, image_id, orig_h, orig_w, want_semantic=False, mask_thr=0.5, score_floor=0.0):
    boxes = pred["boxes"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"]
    if masks.shape[-2:] != (orig_h, orig_w):
        raise ValueError(f"image {image_id}: predicted mask {tuple(masks.shape[-2:])} != image {(orig_h, orig_w)}")
    bin_masks = (masks[:, 0] > mask_thr).to(torch.uint8).cpu().numpy() if len(boxes) else np.zeros((0, orig_h, orig_w), np.uint8)
    bres, sres, sem = [], [], []
    for box, lab, sc, binary in zip(boxes, labels, scores, bin_masks):
        if sc < score_floor or binary.sum() == 0:
            continue
        cat = int(lab) - 1
        if not (0 <= cat < NUM_DISEASE_CLASSES):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        rle = mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("utf-8")
        bres.append({"image_id": int(image_id), "category_id": cat,
                     "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(sc)})
        sres.append({"image_id": int(image_id), "category_id": cat, "segmentation": rle, "score": float(sc)})
        if want_semantic:
            sem.append((binary, int(lab), float(sc)))
    return bres, sres, sem
