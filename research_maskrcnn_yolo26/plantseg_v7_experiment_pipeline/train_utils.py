"""
V2 training utilities.

* Parameter groups (discriminative LR):  backbone 1e-4  <  FPN/RPN 3e-4  <  new heads 1e-3, AdamW wd=1e-4.
  Recipe "flat" reproduces the V1 style (single LR 3e-4, first 4 YOLO layers frozen) and is used by the early
  Tier-2 ablation arms; recipe "staged" = discriminative LR + staged unfreezing; "disc" = Tier 1
  (pretrained heads at the FPN/RPN LR, new predictors at the head LR, no staging).
* Linear warm-up (WARMUP_ITERS) followed by cosine decay  eta_t = eta_min + 0.5(eta_0-eta_min)(1+cos(pi t/T)).
* Staged unfreezing (never resets the optimizer):
      Stage A epochs 1-5   : YOLO backbone frozen (BN frozen), train FPN/RPN/RoI heads/ProtoNet/coeff head
      Stage B epochs 6-20  : upper YOLO stages (> C3) unfrozen
      Stage C epochs 21+   : all backbone stages trainable, backbone LR x0.5
* Model selection  theta* = argmax val Mask AP.  The TEST split is never touched here.
"""
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import V2_ROOT, get_device, SEED
from dataset_v7 import PlantSegCOCODataset, collate_fn, to_float_image
from models import (create_variant, describe_pretraining, print_param_tree, count_params,
                    YOLO26BackboneFPN)
from utils import seed_everything, ensure_dirs
from validation import evaluate_torchvision

LR = {"backbone": 1e-4, "fpn_rpn": 3e-4, "head": 1e-3, "flat": 3e-4}
WEIGHT_DECAY = 1e-4
WARMUP_ITERS = 800
ETA_MIN_RATIO = 0.01
STAGE_B_START, STAGE_C_START = 6, 21
STAGE_C_BACKBONE_MULT = 0.5
GRAD_CLIP = 10.0


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")


def group_name(name: str, recipe: str, variant: str) -> str:
    if recipe == "flat":
        return "flat"
    if name.startswith(("backbone.layers.", "backbone.body.")):
        return "backbone"
    if name.startswith(("backbone.fpn.", "rpn.")):
        return "fpn_rpn"
    if recipe == "disc" and ("predictor" not in name or "conv5_mask" in name):   # pretrained COCO heads
        return "fpn_rpn"
    return "head"


def build_optimizer(model, recipe: str):
    groups = {}
    for n, p in model.named_parameters():
        groups.setdefault(group_name(n, recipe, getattr(model, "variant_name", "")), []).append(p)
    pg = [{"params": ps, "lr": LR[g], "initial_lr": LR[g], "group": g} for g, ps in groups.items()]
    return AdamW(pg, weight_decay=WEIGHT_DECAY), {g: len(ps) for g, ps in groups.items()}


def lr_factor(it, total_iters, group, epoch, recipe):
    warm = min(1.0, (it + 1) / WARMUP_ITERS)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(it, total_iters) / max(1, total_iters)))
    f = warm * (ETA_MIN_RATIO + (1.0 - ETA_MIN_RATIO) * cos)
    if recipe == "staged" and group == "backbone" and epoch >= STAGE_C_START:
        f *= STAGE_C_BACKBONE_MULT
    return f


def stage_for_epoch(epoch):
    return "A" if epoch < STAGE_B_START else ("B" if epoch < STAGE_C_START else "C")


def grad_norm(module):
    s = 0.0
    for p in module.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(s)


def mask_head_modules(model):
    """(proto_net, coeff_head) for Tier 3, else (None, None)."""
    ph = getattr(getattr(model, "roi_heads", None), "proto_mask_head", None)
    return (ph.proto_net, ph.coeff_head) if ph is not None else (None, None)


def prepare_batch(images, targets, device):
    images = [to_float_image(x.to(device, non_blocking=True)) for x in images]
    targets = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()
                if k != "orig_size"} for t in targets]
    keep = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
    return [images[i] for i in keep], [targets[i] for i in keep]


def train_variant(name, epochs=75, batch_size=4, out_root=None, val_every=2, max_train_iters=None,
                  num_workers=5, mask_loss_weight=None, rfs_t=None, resume=False, model_overrides=None,
                  max_val_images=None):
    seed_everything(SEED)
    device = get_device()
    out = Path(out_root) if out_root else V2_ROOT / name
    ensure_dirs(out)
    log = Tee(out / "train.log")
    overrides = dict(model_overrides or {})
    if mask_loss_weight is not None:
        overrides["mask_loss_weight"] = mask_loss_weight
    model = create_variant(name, log=log, **overrides).to(device)
    recipe = model.variant_cfg["recipe"]
    log(f"=== variant {name} | recipe={recipe} | cfg={model.variant_cfg}")
    log("pretraining:", json.dumps(describe_pretraining(model), default=str))
    print_param_tree(model, log)
    (out / "model_info.json").write_text(json.dumps(
        {"variant": name, "cfg": model.variant_cfg, "params_M": count_params(model) / 1e6,
         "pretraining": describe_pretraining(model)}, indent=2, default=str))

    train_ds = PlantSegCOCODataset("train")
    log(f"train images={len(train_ds)} (excluded flagged geometry={train_ds.n_excluded}) | seed={SEED}")
    rfs_idx = train_ds.repeat_indices(rfs_t, SEED) if rfs_t else None
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
                     persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None)
    if rfs_idx is None:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    else:
        from torch.utils.data import Subset
        log(f"repeat-factor sampling t={rfs_t}: {len(rfs_idx)} samples/epoch")
        train_loader = DataLoader(Subset(train_ds, rfs_idx), shuffle=True, **loader_kw)

    opt, gsizes = build_optimizer(model, recipe)
    log("param groups (tensors):", gsizes, "| LR:", {g["group"]: g["lr"] for g in opt.param_groups})
    iters_per_epoch = len(train_loader)
    total_iters = epochs * iters_per_epoch
    start_epoch, best_ap, history, it = 1, -1.0, [], 0
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer"])
        start_epoch, best_ap, history, it = ck["epoch"] + 1, ck["best_ap"], ck["history"], ck["it"]
        log(f"resumed from epoch {ck['epoch']}")

    for epoch in range(start_epoch, epochs + 1):
        if recipe == "staged" and isinstance(model.backbone, YOLO26BackboneFPN):
            model.backbone.set_stage(stage_for_epoch(epoch))
            log(f"epoch {epoch}: stage {stage_for_epoch(epoch)} | trainable params "
                f"{count_params(model, True) / 1e6:.2f} M")
        model.train()
        run, n, comp = 0.0, 0, {}
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"{name} ep{epoch}/{epochs}", leave=False)
        for images, targets in pbar:
            images, targets = prepare_batch(images, targets, device)
            if not images:
                continue
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_factor(it, total_iters, g["group"], epoch, recipe)
            opt.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at it={it}: { {k: float(v) for k, v in loss_dict.items()} }")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            run += float(loss.item()); n += 1; it += 1
            for k, v in loss_dict.items():
                comp[k] = comp.get(k, 0.0) + float(v.item())
            pbar.set_postfix(loss=f"{run / n:.3f}")
            if max_train_iters and it >= max_train_iters:
                break
        rec = {"epoch": epoch, "train_loss": run / max(1, n), "loss_components": {k: v / max(1, n) for k, v in comp.items()},
               "epoch_time_s": time.time() - t0, "stage": stage_for_epoch(epoch) if recipe == "staged" else "-"}
        if epoch % val_every == 0 or epoch == epochs or (max_train_iters and it >= max_train_iters):
            vm = evaluate_torchvision(model, "val", device=device, max_images=max_val_images)
            rec["val_metrics"] = vm
            rec["val_Mask_AP"] = vm.get("Mask_AP", 0.0)
            log(f"epoch {epoch} loss={rec['train_loss']:.4f} val MaskAP={rec['val_Mask_AP']:.3f} "
                f"AP50={vm.get('Mask_AP50', 0):.2f} AP75={vm.get('Mask_AP75', 0):.2f} BoxAP={vm.get('Box_AP', 0):.3f}")
            if rec["val_Mask_AP"] > best_ap:
                best_ap = rec["val_Mask_AP"]
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_Mask_AP": best_ap,
                            "variant": name}, out / "best.pt")
                log(f"[BEST] {name}: val Mask AP={best_ap:.3f} @ epoch {epoch}")
        else:
            log(f"epoch {epoch} loss={rec['train_loss']:.4f} ({rec['epoch_time_s']:.0f}s)")
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model_state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                    "best_ap": best_ap, "history": history, "it": it, "variant": name}, out / "last.pt")
        if max_train_iters and it >= max_train_iters:
            break
    return history


# Backward-compatible name
def train_torchvision_tier(tier, **kw):
    return train_variant({1: "t1", 2: "t2_e", 3: "t3_c"}[tier], **kw)
