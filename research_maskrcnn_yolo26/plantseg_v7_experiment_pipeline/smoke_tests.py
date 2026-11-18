"""
First-execution diagnostics (Part 30).  TRAIN + VAL only; the test split is locked.

  python smoke_tests.py --stage counts   # build every tier, parameter trees, pyramid check, transfer log
  python smoke_tests.py --stage geom     # dataset/mask-projection geometric self-consistency
  python smoke_tests.py --stage t2       # Tier-2 (T2-E recipe) 30-iteration smoke train + tiny val eval
  python smoke_tests.py --stage t3       # Tier-3 overfit/debug (default 500 its on 16 TRAIN images)
  python smoke_tests.py --stage fig      # 8-val-image debug figure from the Tier-3 debug checkpoint
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import train_utils as TU
from config import V2_ROOT, V2_AUDITS, V2_DEBUG, get_device, SEED
from dataset_v7 import PlantSegCOCODataset, collate_fn, to_float_image, polygons_to_mask
from models import (create_variant, print_param_tree, count_params, describe_pretraining, probe_pyramid,
                    yolo_backbone_layers)
from utils import seed_everything

OUT = V2_ROOT / "smoke"
OUT.mkdir(parents=True, exist_ok=True)


def stage_counts():
    rep = {}
    lines = []
    log = lambda *a: (lines.append(" ".join(map(str, a))), print(*a, flush=True))
    for name in ["t1", "t2_a", "t2_b", "t2_c", "t2_d", "t2_e", "t3_b", "t3_c", "t3_k16", "t3_k64"]:
        log(f"\n===== {name} =====")
        m = create_variant(name, log=log)
        print_param_tree(m, log)
        rep[name] = {"params_M": count_params(m) / 1e6,
                     "children_M": {n: count_params(c) / 1e6 for n, c in m.named_children()},
                     "pretraining": describe_pretraining(m)}
        if hasattr(m, "roi_heads"):
            rep[name]["roi_heads_children_M"] = {n: count_params(c) / 1e6 for n, c in m.roi_heads.named_children()}
        # no unused heavy conventional heads inside Tier 3
        if name.startswith("t3_c") or name.startswith("t3_k"):
            names = [n for n, _ in m.named_modules()]
            rep[name]["has_conventional_mask_head"] = any("mask_head" in n and "proto" not in n for n in names)
            rep[name]["has_mask_predictor"] = any("mask_predictor" in n for n in names)
        del m
    (OUT / "param_counts.json").write_text(json.dumps(rep, indent=2, default=str))
    (OUT / "param_counts.log").write_text("\n".join(lines))
    # pyramid consistency between probe script and model
    probe = json.loads((V2_AUDITS / "yolo26_stride_map.json").read_text())["selected_last_at_stride"]
    m = create_variant("t2_b", log=lambda *a: None)
    print("model pyramid:", m.backbone.pyramid_info, "\nprobe  pyramid:",
          {k: (v["idx"], v["channels"]) for k, v in probe.items()})


def stage_geom(n=12):
    """Dataset sanity + mask-projection round trip: GT -> RoIAlign(28x28) at its own box -> paste back -> IoU."""
    from torchvision.models.detection.roi_heads import project_masks_on_boxes, paste_masks_in_image
    ds = PlantSegCOCODataset("train", train=False, load_masks=True)
    ious, aspects = [], []
    for i in np.random.RandomState(0).choice(len(ds), 60, replace=False):
        img, t = ds[int(i)]
        if len(t["boxes"]) == 0:
            continue
        H, W = img.shape[-2:]
        assert t["masks"].shape[-2:] == (H, W)
        k = min(len(t["boxes"]), 8)
        tg = project_masks_on_boxes(t["masks"][:k], t["boxes"][:k], torch.arange(k), 28)       # k,28,28 (soft)
        pasted = paste_masks_in_image(tg[:, None], t["boxes"][:k], (H, W))[:, 0] > 0.5
        gt = t["masks"][:k].bool()
        inter = (pasted & gt).flatten(1).sum(1).float(); uni = (pasted | gt).flatten(1).sum(1).float()
        ious += (inter / uni.clamp(min=1)).tolist()
        aspects.append(W / H)
    res = {"mask_project_paste_roundtrip_IoU_mean": float(np.mean(ious)),
           "mask_project_paste_roundtrip_IoU_min": float(np.min(ious)), "n_instances": len(ious)}
    # aspect ratio preservation through the RCNN transform
    m = create_variant("t3_c", log=lambda *a: None)
    m.eval()
    chk = []
    for i in np.random.RandomState(1).choice(len(ds), 30, replace=False):
        img, t = ds[int(i)]
        il, _ = m.transform([to_float_image(img)], None)
        h, w = il.image_sizes[0]
        H, W = img.shape[-2:]
        s = min(640 / min(H, W), 1024 / max(H, W))
        chk.append((abs(w / h - W / H) / (W / H), abs(h - round(H * s)) <= 1 or abs(w - round(W * s)) <= 1))
    res["aspect_rel_err_max"] = float(max(c[0] for c in chk))
    res["scale_formula_ok_frac"] = float(np.mean([c[1] for c in chk]))
    print(json.dumps(res, indent=2))
    (OUT / "geometry_selfcheck.json").write_text(json.dumps(res, indent=2))


def stage_t2(iters=30):
    TU.WARMUP_ITERS = 10
    t0 = time.time()
    # reuse the real training entry point: 1 "epoch" cut at `iters`, evaluate on 20 val images for smoke test
    hist = TU.train_variant("t2_e", epochs=1, batch_size=4, out_root=OUT / "t2_smoke", max_train_iters=iters,
                            val_every=1, num_workers=4, max_val_images=20)
    print("smoke time", time.time() - t0)


def entropy(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def stage_t3(iters=500, n_img=16, bs=4, variant="t3_c", k=None):
    seed_everything(SEED)
    TU.WARMUP_ITERS = 50
    device = get_device()
    over = {"k": k} if k else {}
    model = create_variant(variant, **over).to(device)
    print_param_tree(model)
    ds = PlantSegCOCODataset("train", train=True)
    ds.augment = False                                      # debug-overfit: deterministic
    # fixed 16 TRAIN images: deterministic, small-ish images with >=2 instances
    cand = [i for i, im in enumerate(ds.images)
            if im["height"] * im["width"] < 1_000_000 and len(ds.by_image.get(im["id"], [])) >= 2]
    sel = np.random.RandomState(SEED).choice(cand, n_img, replace=False).tolist()
    loader = DataLoader(Subset(ds, sel), batch_size=bs, shuffle=True, num_workers=3, collate_fn=collate_fn)
    opt, groups = TU.build_optimizer(model, model.variant_cfg["recipe"])
    model.backbone.set_stage("A")
    proto, coeff = TU.mask_head_modules(model)
    log, it = [], 0
    t0 = time.time()
    abort = None
    while it < iters and abort is None:
        model.train()
        for images, targets in loader:
            images, targets = TU.prepare_batch(images, targets, device)
            if not images:
                continue
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * TU.lr_factor(it, iters, g["group"], 1, "staged")
            opt.zero_grad(set_to_none=True)
            ld = model(images, targets)
            loss = sum(ld.values())
            if not torch.isfinite(loss):
                abort = f"NaN/inf loss at it {it}"; break
            loss.backward()
            gp, gc = TU.grad_norm(proto), TU.grad_norm(coeff)           # BEFORE clipping
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], TU.GRAD_CLIP)
            opt.step()
            d = dict(model.roi_heads.last_diag)
            rec = {"it": it, "loss": float(loss), **{k_: float(v) for k_, v in ld.items()},
                   "proto_grad": gp, "coeff_grad": gc, **d}
            log.append(rec)
            if it % 25 == 0 or it == iters - 1:
                print(f"it {it:4d} loss={rec['loss']:.3f} mask={rec['loss_mask']:.4f} "
                      f"proto grad:{gp:.3e} coeff grad:{gc:.3e} mask logit min/max: "
                      f"{d.get('logit_min', float('nan')):.2f}/{d.get('logit_max', float('nan')):.2f} "
                      f"positive RoIs:{d.get('num_positive')} pred_fg={d.get('pred_fg_ratio', float('nan')):.3f} "
                      f"tgt_fg={d.get('target_fg_ratio', float('nan')):.3f}", flush=True)
            if it > 5 and (gp == 0.0 or gc == 0.0):
                abort = f"zero gradient at it {it}: proto={gp} coeff={gc}"
            if d.get("num_positive", 1) > 0 and it > 100 and (d["pred_fg_ratio"] in (0.0, 1.0)):
                abort = f"degenerate all-{'one' if d['pred_fg_ratio'] == 1.0 else 'zero'} masks at it {it}"
            it += 1
            if abort or it >= iters:
                break
    ok = [r for r in log if "logit_std" in r]
    first, last = ok[: max(1, len(ok) // 10)], ok[-max(1, len(ok) // 10):]
    mean = lambda rs, k_: float(np.mean([r[k_] for r in rs]))
    tgt = float(np.mean([r["target_fg_ratio"] for r in ok]))
    summary = {
        "variant": variant, "iterations_run": it, "abort_reason": abort, "wall_time_s": time.time() - t0,
        "images": n_img, "loss_mask_first10pct": mean(first, "loss_mask"), "loss_mask_last10pct": mean(last, "loss_mask"),
        "loss_mask_decreased": mean(last, "loss_mask") < mean(first, "loss_mask"),
        "const_predictor_BCE_baseline(entropy of mean target fg)": entropy(tgt),
        "proto_grad_min/mean/max": [min(r["proto_grad"] for r in ok), mean(ok, "proto_grad"), max(r["proto_grad"] for r in ok)],
        "coeff_grad_min/mean/max": [min(r["coeff_grad"] for r in ok), mean(ok, "coeff_grad"), max(r["coeff_grad"] for r in ok)],
        "logit_std_last10pct": mean(last, "logit_std"), "logit_min_last": last[-1]["logit_min"], "logit_max_last": last[-1]["logit_max"],
        "pred_fg_ratio_last10pct": mean(last, "pred_fg_ratio"), "target_fg_ratio_mean": tgt,
        "positive_RoIs_per_iter_min/mean": [min(r["num_positive"] for r in ok), mean(ok, "num_positive")],
        "mask_shape_matches_roi_target_all_iters": all(r["shape_ok"] for r in ok),
        "all_losses_finite": all(math.isfinite(r["loss"]) for r in log),
    }
    print(json.dumps(summary, indent=2))
    (OUT / f"{variant}_overfit_log.json").write_text(json.dumps(log))
    (OUT / f"{variant}_overfit_summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"model_state_dict": model.state_dict(), "variant": variant, "k": model.k}, OUT / f"{variant}_debug.pt")
    return summary


def stage_fig(variant="t3_c", n=8):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.ops import box_iou
    from torchvision.models.detection.roi_heads import paste_masks_in_image
    device = get_device()
    model = create_variant(variant).to(device)
    ck = torch.load(OUT / f"{variant}_debug.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    ds = PlantSegCOCODataset("val", train=False, load_masks=True)
    cand = [i for i, im in enumerate(ds.images) if len(ds.by_image.get(im["id"], [])) >= 2
            and im["height"] * im["width"] < 1_500_000]
    sel = cand[:: max(1, len(cand) // n)][:n]
    cols = ["input", "GT polygon masks", "RPN proposals (top 40)", "prototype basis (4 of K)",
            "assembled masks @ positive RoIs", "final predictions"]
    fig, axes = plt.subplots(n, 6, figsize=(21, 3.2 * n))
    rng = np.random.RandomState(0)
    palette = rng.rand(64, 3)

    def overlay(img, masks):
        o = img.copy()
        for j, m in enumerate(masks):
            c = palette[j % 64]
            o[m] = 0.5 * o[m] + 0.5 * c
        return o

    with torch.no_grad():
        for r, i in enumerate(sel):
            img, t = ds[i]
            x = to_float_image(img).to(device)
            tgt = {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device), "masks": t["masks"].to(device)}
            il, tg = model.transform([x], [tgt])
            hh, ww = il.image_sizes[0]
            feats = model.backbone(il.tensors)
            props, _ = model.rpn(il, feats)
            rgb = il.tensors[0, :, :hh, :ww].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            gtm = tg[0]["masks"].bool().cpu().numpy()
            gtb = tg[0]["boxes"]
            # positive RoIs by IoU>=0.5 with GT (same rule as training samples)
            iou = box_iou(props[0], gtb)
            mx, _ = iou.max(1)
            pos = props[0][mx >= 0.5][:8]
            rh = model.roi_heads
            h_pos = rh.box_head(rh.box_roi_pool(feats, [pos], il.image_sizes))
            logits, protomap, _ = rh.proto_mask_head(feats[rh.proto_key], h_pos, [pos], il.image_sizes)
            asm = paste_masks_in_image(logits.sigmoid()[:, None], pos, (hh, ww))[:, 0] > 0.5
            dets, _ = rh(feats, props, il.image_sizes)
            d = dets[0]
            top = d["scores"].argsort(descending=True)[:5]
            fm = paste_masks_in_image(d["masks"][top], d["boxes"][top], (hh, ww))[:, 0] > 0.5
            axes[r, 0].imshow(rgb)
            axes[r, 1].imshow(overlay(rgb, gtm))
            axes[r, 2].imshow(rgb)
            for b in props[0][:40].cpu().numpy():
                axes[r, 2].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="yellow", lw=0.6))
            for b in gtb.cpu().numpy():
                axes[r, 2].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.2))
            pm = protomap[0].float().cpu().numpy()                      # K,Hp,Wp
            sel_k = np.argsort(-pm.reshape(pm.shape[0], -1).std(1))[:4]
            tile = np.zeros((2 * pm.shape[1], 2 * pm.shape[2]))
            for q, kk in enumerate(sel_k):
                ch = pm[kk]; ch = (ch - ch.min()) / (np.ptp(ch) + 1e-6)
                tile[(q // 2) * pm.shape[1]:(q // 2 + 1) * pm.shape[1], (q % 2) * pm.shape[2]:(q % 2 + 1) * pm.shape[2]] = ch
            axes[r, 3].imshow(tile, cmap="magma")
            axes[r, 4].imshow(overlay(rgb, asm.cpu().numpy()))
            for b in pos.cpu().numpy():
                axes[r, 4].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="cyan", lw=0.8))
            axes[r, 5].imshow(overlay(rgb, fm.cpu().numpy()))
            axes[r, 5].set_title(f"n_det(score>=.001)={len(d['scores'])} max={float(d['scores'].max()) if len(d['scores']) else 0:.2f}", fontsize=8)
            for c in range(6):
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(cols[c] if c != 5 else cols[c] + "\n" + axes[r, c].get_title(), fontsize=10)
    fig.suptitle(f"DEBUG ONLY ({variant}: weights from {ck.get('variant')} {int(1)} short TRAIN-subset overfit run; "
                 "val images used only to check mask-assembly geometry) - not for the paper", fontsize=11)
    plt.tight_layout()
    V2_DEBUG.mkdir(parents=True, exist_ok=True)
    p = V2_DEBUG / f"{variant}_mask_assembly_debug.png"
    plt.savefig(p, dpi=60)
    print("saved", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["counts", "geom", "t2", "t3", "fig"])
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--variant", default="t3_c")
    a = ap.parse_args()
    if a.stage == "counts": stage_counts()
    elif a.stage == "geom": stage_geom()
    elif a.stage == "t2": stage_t2(a.iters or 30)
    elif a.stage == "t3": stage_t3(a.iters or 500, variant=a.variant)
    elif a.stage == "fig": stage_fig(a.variant)
