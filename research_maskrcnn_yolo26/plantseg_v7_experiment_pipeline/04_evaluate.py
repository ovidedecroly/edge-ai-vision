"""
04_evaluate.py (V2): COCO instance metrics + semantic metrics computed from a confusion matrix.

  * default split = val.  split=test is refused unless config.ALLOW_TEST_EVAL is True (frozen architecture
    approved by the user).
  * Semantic projection: pixels are labelled by the highest-score instance covering them (score >= --sem-score-thr),
    background = 0.  Confusion matrix over 116 labels (bg + 115 diseases);
        IoU_c = TP_c/(TP_c+FP_c+FN_c),   mIoU = mean over the 115 disease classes (background excluded),
        Acc_c = TP_c/(TP_c+FN_c),        mean_class_accuracy = mean_c Acc_c   (NOT pixel accuracy),
        pixel_accuracy reported separately (foreground+background pixels).
  * No resize repair of masks anywhere: shape mismatches raise.
"""
import argparse
import contextlib
import io
import json

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import V2_ROOT, V2_VALRES, NUM_DISEASE_CLASSES, get_device, PREPARED_ROOT, OUTPUT_ROOT, resolve_dataset_dir
from dataset_v7 import PlantSegCOCODataset, collate_fn, guard_split, to_float_image
from eval_utils import build_coco, predictions_to_coco, gt_semantic_map
from metrics import coco_eval_from_results, semantic_confusion_update, semantic_metrics
from models import create_variant
from utils import torch_checkpoint_state, ensure_dirs

NC = NUM_DISEASE_CLASSES + 1


def finalize(out, conf):
    sem = semantic_metrics(conf)
    pc_iou = np.asarray(sem["per_class_IoU"], dtype=float)
    pc_acc = np.asarray(sem["per_class_Acc"], dtype=float)
    out["mIoU_with_background"] = sem["mIoU"]
    out["mIoU"] = float(np.nanmean(pc_iou[1:]))
    out["mean_class_accuracy"] = float(np.nanmean(pc_acc[1:]))          # mean class accuracy (recall), NOT pixel acc.
    out["pixel_accuracy_all_pixels"] = sem["pixel_accuracy"]
    out["n_disease_classes_in_mean"] = int(np.isfinite(pc_iou[1:]).sum())
    out["per_class_IoU"] = pc_iou[1:].tolist()
    out["per_class_mean_class_acc"] = pc_acc[1:].tolist()
    return out


@torch.no_grad()
def evaluate_variant(variant, split, ckpt=None, sem_thr=0.05):
    guard_split(split)
    device = get_device()
    model = create_variant(variant, log=lambda *a: None).to(device)
    ckpt = ckpt or V2_ROOT / variant / "best.pt"
    model.load_state_dict(torch_checkpoint_state(ckpt, device), strict=True)
    model.eval()
    ds = PlantSegCOCODataset(split, train=False, load_masks=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=collate_fn)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = build_coco(split)
    box_res, seg_res = [], []
    conf = np.zeros((NC, NC), dtype=np.int64)
    for images, targets in loader:
        t = targets[0]
        pred = model([to_float_image(images[0].to(device))])[0]
        h, w = [int(x) for x in t["orig_size"]]
        iid = int(t["image_id"].item())
        b, s, sem_items = predictions_to_coco(pred, iid, h, w, want_semantic=True)
        box_res.extend(b); seg_res.extend(s)
        pm = np.zeros((h, w), dtype=np.int32)
        for mask, lab, score in sorted(sem_items, key=lambda x: x[2]):
            if score >= sem_thr:
                pm[mask.astype(bool)] = lab
        semantic_confusion_update(conf, gt_semantic_map(coco, iid, h, w), pm, NC)
    with contextlib.redirect_stdout(io.StringIO()):
        out = coco_eval_from_results(coco, box_res, seg_res)
    out["semantic_score_threshold"] = sem_thr
    return finalize(out, conf)


@torch.no_grad()
def evaluate_yolo(split, weights, sem_thr=0.05):
    """Tier-4 reference. retina_masks=True so masks come back on the ORIGINAL image grid (the V1 path resized
    letterboxed network-grid masks to (w,h), which is geometrically wrong)."""
    from ultralytics import YOLO
    from pycocotools import mask as mask_utils
    guard_split(split)
    ds_root = resolve_dataset_dir()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = build_coco(split)
    model = YOLO(str(weights))
    box_res, seg_res = [], []
    conf = np.zeros((NC, NC), dtype=np.int64)
    for info in coco.dataset["images"]:
        p = ds_root / "images" / split / info["file_name"]
        h, w = info["height"], info["width"]
        r = model.predict(str(p), imgsz=640, conf=0.001, iou=0.7, max_det=100, retina_masks=True, verbose=False)[0]
        assert tuple(r.orig_shape) == (h, w), (p, r.orig_shape, (h, w))
        pm = np.zeros((h, w), dtype=np.int32)
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy(); scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
            for bx, sc, cid, m in zip(boxes, scores, labels, masks):
                if m.sum() == 0:
                    continue
                rle = mask_utils.encode(np.asfortranarray(m)); rle["counts"] = rle["counts"].decode()
                box_res.append({"image_id": int(info["id"]), "category_id": int(cid),
                                "bbox": [float(bx[0]), float(bx[1]), float(bx[2] - bx[0]), float(bx[3] - bx[1])], "score": float(sc)})
                seg_res.append({"image_id": int(info["id"]), "category_id": int(cid), "segmentation": rle, "score": float(sc)})
            for i in np.argsort(scores):
                if scores[i] >= sem_thr:
                    pm[masks[i].astype(bool)] = int(labels[i]) + 1
        semantic_confusion_update(conf, gt_semantic_map(coco, info["id"], h, w), pm, NC)
    with contextlib.redirect_stdout(io.StringIO()):
        out = coco_eval_from_results(coco, box_res, seg_res)
    out["semantic_score_threshold"] = sem_thr
    return finalize(out, conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="t1,t2_*,t3_*, or tier4")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sem-score-thr", type=float, default=0.05)
    a = ap.parse_args()
    ensure_dirs(V2_VALRES)
    if a.variant == "tier4":
        w = a.ckpt or max((OUTPUT_ROOT / "yolo_runs").rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime)
        out = evaluate_yolo(a.split, w, a.sem_score_thr)
    else:
        out = evaluate_variant(a.variant, a.split, a.ckpt, a.sem_score_thr)
    path = V2_VALRES / f"{a.variant}_{a.split}_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("per_class")}, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    main()
