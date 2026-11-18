import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_BOX_KEYS = ["Box_AP", "Box_AP50", "Box_AP75", "Box_APs", "Box_APm", "Box_APl", "Box_AR100"]
_MASK_KEYS = ["Mask_AP", "Mask_AP50", "Mask_AP75", "Mask_APs", "Mask_APm", "Mask_APl", "Mask_AR100"]
_STAT_IDX = [0, 1, 2, 3, 4, 5, 8]


def _run(coco_gt, results, iou_type, keys):
    if not results:   # no detections at all -> explicit zeros (never silently missing keys)
        return {k: 0.0 for k in keys}
    dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, dt, iou_type)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    # COCOeval returns -1 when a size bucket has no GT; keep -1 -> reported as NaN-safe 0? keep faithful: NaN.
    return {k: (float(s[i] * 100) if s[i] >= 0 else float("nan")) for k, i in zip(keys, _STAT_IDX)}


def coco_eval_from_results(coco_gt: COCO, bbox_results, segm_results):
    out = {}
    out.update(_run(coco_gt, bbox_results, "bbox", _BOX_KEYS))
    out.update(_run(coco_gt, segm_results, "segm", _MASK_KEYS))
    return out


def semantic_confusion_update(conf, gt, pred, ncls):
    valid = (gt >= 0) & (gt < ncls)
    x = ncls * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    conf += np.bincount(x, minlength=ncls * ncls).reshape(ncls, ncls)


def semantic_metrics(conf):
    """conf[i,j] = #pixels with GT class i predicted as class j (index 0 = background).
       IoU_c = TP_c/(TP_c+FP_c+FN_c); mIoU = mean_c IoU_c;
       Acc_c = TP_c/(TP_c+FN_c) (class recall);  mean_class_accuracy = mean_c Acc_c  (NOT pixel accuracy);
       pixel_accuracy = sum_c TP_c / sum of all pixels.
       Classes with an empty denominator (absent in GT and prediction) are excluded from the mean (NaN)."""
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    gt_sum = conf.sum(1)
    pred_sum = conf.sum(0)
    denom = gt_sum + pred_sum - tp
    iou = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
    acc = np.divide(tp, gt_sum, out=np.full_like(tp, np.nan), where=gt_sum > 0)
    return {
        "mIoU": float(np.nanmean(iou) * 100),
        "mAcc": float(np.nanmean(acc) * 100),
        "mean_class_accuracy": float(np.nanmean(acc) * 100),
        "pixel_accuracy": float(tp.sum() / max(1.0, conf.sum()) * 100),
        "n_classes_in_mean": int(np.isfinite(iou).sum()),
        "per_class_IoU": (iou * 100).tolist(),
        "per_class_Acc": (acc * 100).tolist(),
    }
