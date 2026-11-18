"""
Train-split-ONLY anchor statistics.

* Boxes come from the geometry-corrected COCO json (coco_geomfix), restricted to the TRAINING view
  (images flagged by audit_geometry as inconsistent are excluded, exactly as in dataset_v7.py).
* "Effective training scale": every box is multiplied by the aspect-preserving factor
        s = min(MIN_SIZE / min(H,W), MAX_SIZE / max(H,W))      (H,W = oriented image size)
  which is precisely what torchvision GeneralizedRCNNTransform applies at nominal scale.
* Reports quantiles of sqrt(w*h) and of aspect ratio (w/h and h/w).
* Derives candidate RPN anchors from the observed distribution (no hand-assumed sizes) and compares the
  candidates by shape-coverage (best centred-anchor IoU per GT box).

Output: runs/plantseg_v7_v2/anchor_statistics.json (+ anchors/ copy)
"""
import json
import numpy as np

from config import V2_ROOT, V2_ANCHORS, V2_AUDITS, MIN_SIZE, MAX_SIZE
from dataset_v7 import load_geomfix_coco, load_flags

QS = [5, 10, 25, 50, 75, 90, 95]


def effective_boxes(split="train"):
    coco = load_geomfix_coco(split)
    flags = load_flags()[split]
    excl = set(flags["flagged_inconsistent"]) if split == "train" else set()
    info = {im["id"]: im for im in coco["images"]}
    s_list, r_list, cls = [], [], []
    for a in coco["annotations"]:
        im = info[a["image_id"]]
        if im["file_name"] in excl:
            continue
        H, W = im["height"], im["width"]
        s = min(MIN_SIZE / min(H, W), MAX_SIZE / max(H, W))
        x, y, w, h = a["bbox"]
        x2, y2 = min(W, x + w), min(H, y + h)
        x, y = max(0, x), max(0, y)
        w, h = (x2 - x) * s, (y2 - y) * s
        if w <= 1 or h <= 1:
            continue
        s_list.append(float(np.sqrt(w * h)))
        r_list.append(float(w / h))
        cls.append(int(a["category_id"]))
    return np.array(s_list), np.array(r_list), np.array(cls)


def shape_coverage(sz, ar_hw, sizes_per_level, ratios_per_level):
    """Best IoU between each GT (w,h centred) and every anchor shape (centred)."""
    s = sz[:, None]
    r = ar_hw[:, None]             # w/h
    gw, gh = s * np.sqrt(r), s / np.sqrt(r)
    aw, ah = [], []
    for sizes, ratios in zip(sizes_per_level, ratios_per_level):
        for a in sizes:
            for hw in ratios:      # torchvision: ratio = h/w
                aw.append(a / np.sqrt(hw)); ah.append(a * np.sqrt(hw))
    aw, ah = np.array(aw)[None], np.array(ah)[None]
    inter = np.minimum(gw, aw) * np.minimum(gh, ah)
    iou = inter / (gw * gh + aw * ah - inter)
    best = iou.max(1)
    return {"mean_best_IoU": float(best.mean()), "frac_ge_0.5": float((best >= 0.5).mean()),
            "frac_ge_0.7": float((best >= 0.7).mean())}


def derive_anchors(sz, r_wh, n_levels=5):
    q05, q95 = np.percentile(sz, 5), np.percentile(sz, 95)
    # geometric ladder from Q05 to Q95 across levels, snapped to multiples of 8
    raw = np.geomspace(q05, q95, n_levels)
    sizes = [int(max(8, round(v / 8) * 8)) for v in raw]
    for i in range(1, len(sizes)):                    # strictly increasing
        sizes[i] = max(sizes[i], sizes[i - 1] + 8)
    r_hw = 1.0 / r_wh
    lo, md, hi = np.percentile(r_hw, [10, 50, 90])
    if np.percentile(r_hw, 5) >= 0.4 and np.percentile(r_hw, 95) <= 2.5:
        ratios = (0.5, 1.0, 2.0)
        ratio_rule = "kept {0.5,1,2}: train Q05/Q95 of h/w inside [0.4,2.5]"
    else:
        ratios = tuple(float(round(v, 2)) for v in (lo, md, hi))
        ratio_rule = "derived from train h/w quantiles (Q10, Q50, Q90)"
    return sizes, ratios, ratio_rule


def main():
    sz, r_wh, cls = effective_boxes("train")
    out = {
        "split": "train (training view, flagged images excluded)",
        "n_boxes": int(len(sz)),
        "effective_scale": f"s=min({MIN_SIZE}/min(H,W), {MAX_SIZE}/max(H,W)), nominal scale, aspect preserved",
        "sqrt_wh_quantiles": {f"Q{q:02d}": float(np.percentile(sz, q)) for q in QS},
        "sqrt_wh_mean": float(sz.mean()),
        "aspect_w_over_h_quantiles": {f"Q{q:02d}": float(np.percentile(r_wh, q)) for q in QS},
        "aspect_h_over_w_quantiles": {f"Q{q:02d}": float(np.percentile(1 / r_wh, q)) for q in QS},
        "fraction_size_lt": {str(t): float((sz < t).mean()) for t in (8, 16, 32, 64, 128, 256)},
    }
    sizes5, ratios, rule = derive_anchors(sz, r_wh, 5)
    # a 4-level variant (P3..P6) for the no-P2 ablation arm (T2-A/B uses same stats)
    sizes4, _, _ = derive_anchors(sz, r_wh, 4)
    cand = {
        "V1_default_P3-P6": ([(32,), (64,), (128,), (256,)], [(0.5, 1.0, 2.0)] * 4),
        "pow2_P2-P6_8..128": ([(8,), (16,), (32,), (64,), (128,)], [(0.5, 1.0, 2.0)] * 5),
        "derived_P2-P6": ([(s,) for s in sizes5], [ratios] * 5),
        "derived_P3-P6": ([(s,) for s in sizes4], [ratios] * 4),
    }
    out["candidates"] = {k: {"sizes": [s[0] for s in v[0]], "ratios": list(v[1][0]),
                             "coverage": shape_coverage(sz, r_wh, v[0], v[1])}
                         for k, v in cand.items()}
    out["selected"] = {"sizes_P2_P6": sizes5, "sizes_P3_P6": sizes4, "aspect_ratios_h_over_w": list(ratios),
                       "ratio_rule": rule,
                       "sizes_rule": "geometric ladder Q05..Q95 of sqrt(wh) over the levels, snapped to multiples of 8"}
    # long tail (Part 17) is also train-only
    n = np.bincount(cls, minlength=115)
    nz = n[n > 0]
    out["long_tail_train"] = {"n_classes_with_instances": int(len(nz)), "min": int(nz.min()),
                              "median": float(np.median(nz)), "max": int(nz.max()),
                              "imbalance_ratio": float(nz.max() / nz.min())}
    for d in (V2_ROOT, V2_ANCHORS):
        d.mkdir(parents=True, exist_ok=True)
        (d / "anchor_statistics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
