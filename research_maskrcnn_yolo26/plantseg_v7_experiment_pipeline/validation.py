"""Validation-only COCO evaluation for torchvision-style models (V2). Test split is guarded."""
import contextlib
import io

import torch
from torch.utils.data import DataLoader

from dataset_v7 import PlantSegCOCODataset, collate_fn, guard_split, to_float_image
from eval_utils import build_coco, predictions_to_coco
from metrics import coco_eval_from_results


@torch.no_grad()
def evaluate_torchvision(model, split="val", device=None, max_images=None, num_workers=3, quiet=True):
    """Mask AP@[.5:.95], AP50, AP75, APs/m/l, Box AP/50/75, AR100 (percent)."""
    guard_split(split)
    if device is None:
        device = next(model.parameters()).device
    ds = PlantSegCOCODataset(split, train=False, load_masks=False)   # eval: original res, no GT masks loaded
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    sink = io.StringIO()
    with (contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()):
        coco = build_coco(split)
    was_training = model.training
    model.eval()
    box_results, seg_results = [], []
    for i, (images, targets) in enumerate(loader):
        if max_images and i >= max_images:
            break
        img = to_float_image(images[0].to(device))
        pred = model([img])[0]
        h, w = [int(x) for x in targets[0]["orig_size"]]
        b, s, _ = predictions_to_coco(pred, int(targets[0]["image_id"].item()), h, w)
        box_results.extend(b); seg_results.extend(s)
    with (contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()):
        out = coco_eval_from_results(coco, box_results, seg_results)
    model.train(was_training)
    out["n_images"] = i + 1 if max_images is None else min(max_images, len(ds))
    out["n_detections"] = len(seg_results)
    return out
