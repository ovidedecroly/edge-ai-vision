"""
Geometry audit for PlantSeg v7 (train/val/test).  MEASURES FIRST, then emits a deterministic fix.

For every image compares
  * raw stored pixel dimensions        (PIL, EXIF NOT applied)
  * oriented dimensions                (cv2.imread default => EXIF orientation applied)
  * COCO JSON height/width
  * pycocotools annToMask shape        (== JSON h,w by construction)
  * polygon vertex extents             (which frame do polygons live in?)
and tests the transposition hypothesis explicitly (mask.T vs correct rasterisation).

NO cv2.resize workaround is used anywhere.  Findings (see geometry_audit.json) drive:
  (1) EXIF-axis-swapped images -> polygons live in the EXIF-oriented frame; JSON h/w are the raw (swapped)
      dims.  Fix = rasterise in the oriented frame and write corrected h/w into a V2 copy of the COCO json.
  (2) Images whose polygons run outside the oriented image frame by > OVERSHOOT_TOL are flagged as
      geometrically inconsistent annotations.  They are EXCLUDED FROM TRAINING ONLY (val/test never modified;
      flagged test images are reported, not altered).

Outputs
  runs/plantseg_v7_v2/geometry_audit.json  (+ audits/ copy)
  runs/plantseg_v7_v2/audits/geometry_flags.json          (consumed by dataset_v7.py)
  runs/plantseg_v7_v2/coco_geomfix/annotation_{split}.clean.json  (consumed by dataset/validation)
"""
import json
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from config import PREPARED_ROOT, V2_ROOT, V2_AUDITS, resolve_dataset_dir

SPLITS = ["train", "val", "test"]
EXIF_ORIENTATION = 274
OVERSHOOT_TOL = 1.02          # polygon extent / oriented image extent; > tol  => flagged
GEOMFIX_DIR = V2_ROOT / "coco_geomfix"


def _poly_ok(poly):
    return isinstance(poly, (list, tuple)) and len(poly) >= 6 and len(poly) % 2 == 0 \
        and np.isfinite(np.asarray(poly, dtype=float)).all()


def _rasterise(polys, h, w):
    rles = mask_utils.frPyObjects(polys, h, w)
    return mask_utils.decode(mask_utils.merge(rles))


def _probe(args):
    path, jh, jw, polys = args
    try:
        with Image.open(path) as im:
            raw_w, raw_h = im.size
            exif = im.getexif().get(EXIF_ORIENTATION)
    except FileNotFoundError:
        return {"missing": True}
    cv = cv2.imread(path)  # default flag applies EXIF orientation
    if cv is None:
        return {"missing": True, "unreadable": True}
    ch, cw = cv.shape[:2]
    r = {"missing": False, "raw_hw": (raw_h, raw_w), "cv_hw": (ch, cw), "exif": exif, "json_hw": (jh, jw)}
    good = [p for p in polys if _poly_ok(p)]
    if good:
        r["poly_max_x"] = float(max(max(p[0::2]) for p in good))
        r["poly_max_y"] = float(max(max(p[1::2]) for p in good))
        # transposition test: json-frame raster transposed vs correct oriented-frame raster
        if (jh, jw) == (cw, ch) and ch != cw:
            a = _rasterise(good, jh, jw).T          # annToMask-equivalent, then transposed
            b = _rasterise(good, ch, cw)            # correct (oriented) frame
            u = np.logical_or(a, b).sum()
            r["transpose_iou"] = float(np.logical_and(a, b).sum() / u) if u else 1.0
            a0 = _rasterise(good, jh, jw)
            r["transpose_untransposed_shape_equal"] = bool(a0.shape == b.shape)
    return r


def audit_split(split, ds_dir):
    coco = COCO(str(ds_dir / f"annotation_{split}.json"))   # RAW json as shipped
    jobs, ids = [], []
    malformed = n_ann = 0
    for img_id, info in coco.imgs.items():
        flat = []
        for a in coco.imgToAnns.get(img_id, []):
            n_ann += 1
            seg = a.get("segmentation")
            if not isinstance(seg, list) or len(seg) == 0 or not all(_poly_ok(p) for p in seg):
                malformed += 1
                continue
            flat.extend(seg)
        jobs.append((str(ds_dir / "images" / split / info["file_name"]), info["height"], info["width"], flat))
        ids.append(img_id)
    with Pool(10) as pool:
        res = pool.map(_probe, jobs, chunksize=32)

    c = Counter()
    swap_fix, flagged = [], []
    ex_exif, transpose_ious = [], []
    over_hist = Counter()
    for img_id, r in zip(ids, res):
        info = coco.imgs[img_id]
        c["images"] += 1
        if r["missing"]:
            c["missing_images"] += 1
            continue
        jh, jw = r["json_hw"]; rh, rw = r["raw_hw"]; ch, cw = r["cv_hw"]; exif = r["exif"]
        # --- JSON vs stored/oriented dims
        c["json_exact_match_raw_stored"] += int((rh, rw) == (jh, jw))
        c["json_exact_match_oriented"] += int((ch, cw) == (jh, jw))
        if (ch, cw) == (jw, jh) and ch != cw:
            c["json_wh_swapped_vs_oriented"] += 1
        if (ch, cw) not in ((jh, jw), (jw, jh)):
            c["true_resize_mismatch_json_vs_oriented"] += 1
        if (rh, rw) not in ((jh, jw), (jw, jh)):
            c["true_resize_mismatch_json_vs_raw"] += 1
        if exif not in (None, 1):
            c["exif_orientation_present"] += 1
            ex_exif.append({"file": info["file_name"], "exif": exif, "raw_hw": [rh, rw], "oriented_hw": [ch, cw], "json_hw": [jh, jw]})
        # --- masks requiring axis handling
        if (ch, cw) == (jw, jh) and ch != cw:
            c["annToMask_shape_transposed_vs_oriented"] += 1
            swap_fix.append(info["file_name"])
            if "transpose_iou" in r:
                transpose_ious.append(r["transpose_iou"])
        # --- polygon overshoot in the ORIENTED frame
        if "poly_max_x" in r:
            ov = max(r["poly_max_x"] / cw, r["poly_max_y"] / ch)
            for t in (1.02, 1.05, 1.10, 1.5):
                c[f"polygon_overshoot_gt_{t}"] += int(ov > t)
            if ov > OVERSHOOT_TOL:
                flagged.append({"file": info["file_name"], "overshoot": round(ov, 3), "oriented_hw": [ch, cw]})
    c["annotations_total"] = n_ann
    c["malformed_polygons"] = malformed
    summary = dict(c)
    if transpose_ious:
        summary["transpose_hypothesis_mean_IoU"] = float(np.mean(transpose_ious))
        summary["transpose_hypothesis_note"] = (
            "IoU between annToMask(json frame).T and the correct oriented-frame rasterisation. "
            "Transposition is NOT a rotation, so it is rejected whenever this is << 1.")
    return summary, swap_fix, flagged, ex_exif


def build_geomfix_coco(split, swap_files):
    """Deterministic V2 COCO json: only width/height of EXIF-swapped images are corrected."""
    base = json.loads((PREPARED_ROOT / "coco" / f"annotation_{split}.clean.json").read_text())
    sw = set(swap_files)
    n = 0
    for im in base["images"]:
        if im["file_name"] in sw:
            im["height"], im["width"] = im["width"], im["height"]
            n += 1
    GEOMFIX_DIR.mkdir(parents=True, exist_ok=True)
    (GEOMFIX_DIR / f"annotation_{split}.clean.json").write_text(json.dumps(base))
    return n


def main():
    ds_dir = resolve_dataset_dir()
    V2_AUDITS.mkdir(parents=True, exist_ok=True)
    report = {"dataset_dir": str(ds_dir), "overshoot_tolerance": OVERSHOOT_TOL, "splits": {}}
    flags = {}
    for s in SPLITS:
        print("auditing", s, flush=True)
        summ, swap_fix, flagged, ex = audit_split(s, ds_dir)
        n_fixed = build_geomfix_coco(s, swap_fix)
        summ["geomfix_json_images_corrected"] = n_fixed
        report["splits"][s] = {"counts": summ, "exif_cases": ex, "flagged_polygon_overshoot": flagged}
        flags[s] = {"exif_swap_fixed": swap_fix, "flagged_inconsistent": [f["file"] for f in flagged]}
    tot = Counter()
    for s in SPLITS:
        for k, v in report["splits"][s]["counts"].items():
            if isinstance(v, (int, float)) and not k.startswith("transpose_hypothesis_mean"):
                tot[k] += v
    report["total"] = dict(tot)
    report["policy"] = {
        "image_loading": "cv2.imread default (EXIF orientation applied) for ALL images",
        "mask_rasterisation": "pycocotools frPyObjects at oriented (H,W); no resize; no transpose",
        "coco_gt_json": "runs/plantseg_v7_v2/coco_geomfix (only h/w of EXIF-swapped images corrected)",
        "train_exclusion": f"images with polygon overshoot > {OVERSHOOT_TOL} excluded from TRAINING only",
        "val_test": "never filtered or modified; flagged images are reported",
    }
    (V2_ROOT / "geometry_audit.json").write_text(json.dumps(report, indent=2))
    (V2_AUDITS / "geometry_audit.json").write_text(json.dumps(report, indent=2))
    (V2_AUDITS / "geometry_flags.json").write_text(json.dumps(flags, indent=2))
    print(json.dumps(report["total"], indent=2))
    for s in SPLITS:
        print(s, json.dumps(report["splits"][s]["counts"]))


if __name__ == "__main__":
    main()
