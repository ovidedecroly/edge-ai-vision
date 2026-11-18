"""
V2 models for PlantSeg v7.

Tier 1  : Mask R-CNN ResNet50-FPN (COCO-pretrained control, unchanged architecture).
Tier 2  : YOLO26-backbone Mask R-CNN with (optional) high-resolution P2 feature.
Tier 3  : YOLO26-backbone + RoI-aligned prototype mask head ("ProtoMaskRCNN").

All backbone indices are DISCOVERED by probing the actual YOLO26 graph (no hard-coded P2 index).
Weights/preprocessing facts (logged by `describe_pretraining`):
  Tier 1 : torchvision COCO Mask R-CNN (all components), ImageNet mean/std.
  Tier 2 : YOLO26-seg pretrained backbone; + shape-matched COCO detector modules when transfer=True.
  Tier 3 : same YOLO26 backbone init, same shape-matched COCO RPN/FPN-smoothing transfer; new ProtoNet/coeff head.
YOLO tiers use image_mean=0, image_std=1: the Ultralytics backbone was trained on RGB in [0,1] (V1 silently
fed it ImageNet-normalised input through torchvision's default transform).
"""
import copy
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, project_masks_on_boxes
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from config import YOLO26_WEIGHTS, NUM_MASKRCNN_CLASSES, MIN_SIZE, MAX_SIZE

TRAIN_MIN_SIZES = (544, 576, 608, 640, 672, 704, 736)   # +-15% scale jitter, aspect preserving
EVAL_BOX_SCORE_THRESH = 0.001
EVAL_DETS_PER_IMG = 100

# Train-derived anchors (runs/plantseg_v7_v2/anchor_statistics.json, "selected").
DERIVED_ANCHOR_SIZES_P2_P6 = (16, 32, 72, 168, 400)
DERIVED_ANCHOR_RATIOS = (0.58, 1.02, 2.07)
# "Existing" anchors (V1) and their 5-level extension for the P2 arm (no train statistics used).
DEFAULT_SIZES_P3_P6 = (32, 64, 128, 256)
DEFAULT_SIZES_P2_P6 = (16, 32, 64, 128, 256)
DEFAULT_RATIOS = (0.5, 1.0, 2.0)


def count_params(m: nn.Module, trainable_only=False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


def print_param_tree(model: nn.Module, log=print, prefix=""):
    """Params (M) per child module, recursively one level into roi_heads / backbone."""
    for name, mod in model.named_children():
        log(f"{prefix}{name:<14s} {count_params(mod) / 1e6:8.3f} M")
        if name in ("roi_heads", "backbone"):
            for n2, m2 in mod.named_children():
                log(f"{prefix}  {name}.{n2:<18s} {count_params(m2) / 1e6:8.3f} M")
    log(f"{prefix}TOTAL          {count_params(model) / 1e6:8.3f} M")


# --------------------------------------------------------------------------------------------------
# Transform with native multi-scale (scale-jitter) training, fixed-scale evaluation
# --------------------------------------------------------------------------------------------------
def safe_paste_masks_in_image(masks, boxes, img_shape, padding=1):
    if len(masks) == 0:
        return torch.zeros((0, 1, img_shape[0], img_shape[1]), dtype=torch.uint8)
    from torchvision.models.detection.roi_heads import expand_masks, expand_boxes
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape
    res = []
    for m, b in zip(masks, boxes):
        w = max(int(b[2] - b[0] + 1), 1)
        h = max(int(b[3] - b[1] + 1), 1)
        m_roi = F.interpolate(m.expand((1, 1, -1, -1)), size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
        out_cpu = torch.zeros((im_h, im_w), dtype=torch.uint8)
        x_0, x_1 = max(int(b[0]), 0), min(int(b[2] + 1), im_w)
        y_0, y_1 = max(int(b[1]), 0), min(int(b[3] + 1), im_h)
        if x_1 > x_0 and y_1 > y_0:
            roi_sub = m_roi[(y_0 - b[1]):(y_1 - b[1]), (x_0 - b[0]):(x_1 - b[0])]
            out_cpu[y_0:y_1, x_0:x_1] = roi_sub.to(dtype=torch.uint8, device="cpu")
        res.append(out_cpu)
    return torch.stack(res, dim=0)[:, None]


class RCNNTransform(GeneralizedRCNNTransform):
    """Aspect-preserving resize  s = min(Smin/min(H,W), Smax/max(H,W)).  Training samples Smin from
    `train_min_sizes` (boxes and masks rescaled by torchvision); eval uses min_size[-1] = MIN_SIZE."""

    def __init__(self, min_size, max_size, image_mean, image_std, train_min_sizes=None, **kw):
        super().__init__(min_size, max_size, image_mean, image_std, **kw)
        self.train_min_sizes = tuple(train_min_sizes) if train_min_sizes else None

    def resize(self, image, target=None):
        if self.training and self.train_min_sizes:
            saved = self.min_size
            self.min_size = self.train_min_sizes
            try:
                return super().resize(image, target)
            finally:
                self.min_size = saved
        return super().resize(image, target)

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        from torchvision.models.detection.transform import resize_boxes
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                # Use safe GPU-interpolate-to-CPU-uint8 paste to prevent 6+ GB CUDA VRAM spikes on high-res images
                result[i]["masks"] = safe_paste_masks_in_image(masks, boxes, o_im_s)
        return result


def make_transform(yolo_style: bool, train_min_sizes=TRAIN_MIN_SIZES):
    mean, std = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) if yolo_style else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return RCNNTransform((MIN_SIZE,), MAX_SIZE, mean, std, train_min_sizes=train_min_sizes)


# --------------------------------------------------------------------------------------------------
# YOLO26 backbone + FPN, pyramid indices discovered dynamically
# --------------------------------------------------------------------------------------------------
def yolo_backbone_layers(yolo) -> nn.ModuleList:
    """Sequential backbone = layers up to (excluding) the first layer that belongs to the head
    (first Upsample/Concat or layer reading from a list of earlier layers)."""
    keep = []
    for i, layer in enumerate(yolo.model.model):
        f = getattr(layer, "f", -1)
        if isinstance(f, list) or type(layer).__name__ in ("Concat", "Upsample") or f != -1:
            break
        keep.append(layer)
    return nn.ModuleList(keep)


def probe_pyramid(layers: nn.ModuleList, size=640) -> Dict[int, Tuple[int, int]]:
    """{stride: (layer_idx, channels)} using the LAST (most refined) layer observed at each stride."""
    sel = {}
    was_training = layers.training
    layers.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, size, size)
        for i, layer in enumerate(layers):
            x = layer(x)
            sel[size // x.shape[-1]] = (i, x.shape[1])
    layers.train(was_training)
    return sel


class YOLO26BackboneFPN(nn.Module):
    """C2..C5 (strides 4..32) -> FPN -> P2..P5 + P6 (= max-pool of P5).
    P5 = phi5(C5);  P_l = psi_l[ phi_l(C_l) + Up2(P_{l+1}) ];  P6 = MaxPool(P5)
    (torchvision FeaturePyramidNetwork: phi = 1x1 lateral conv, psi = 3x3 smoothing conv)."""

    def __init__(self, yolo_weights=str(YOLO26_WEIGHTS), out_channels=256, use_p2=True, freeze_first=0):
        super().__init__()
        y = YOLO(yolo_weights)
        self.layers = yolo_backbone_layers(y)
        pyr = probe_pyramid(self.layers)
        self.strides = [4, 8, 16, 32] if use_p2 else [8, 16, 32]
        self.idx = [pyr[s][0] for s in self.strides]
        self.chs = [pyr[s][1] for s in self.strides]
        self.last_idx = max(self.idx)
        self.c3_end = pyr[8][0]                       # layers <= c3_end are the "lower" stages
        self.pyramid_info = {f"stride{s}": {"layer": pyr[s][0], "channels": pyr[s][1]} for s in self.strides}
        self.fpn = FeaturePyramidNetwork(self.chs, out_channels, extra_blocks=LastLevelMaxPool())
        self.out_channels = out_channels
        for i in range(min(freeze_first, len(self.layers))):
            for p in self.layers[i].parameters():
                p.requires_grad = False

    def set_stage(self, stage: str):
        """A: whole YOLO backbone frozen; B: upper stages (> C3) trainable; C: everything trainable."""
        for i, layer in enumerate(self.layers):
            train_it = stage == "C" or (stage == "B" and i > self.c3_end)
            for p in layer.parameters():
                p.requires_grad = train_it

    def train(self, mode=True):
        super().train(mode)
        if mode:                                      # frozen layers keep BN statistics fixed
            for layer in self.layers:
                if not any(p.requires_grad for p in layer.parameters()):
                    layer.eval()
        return self

    def forward(self, x):
        feats = {}
        want = set(self.idx)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i in want:
                feats[i] = out
            if i == self.last_idx:
                break
        return self.fpn(OrderedDict((str(k), feats[i]) for k, i in enumerate(self.idx)))


# --------------------------------------------------------------------------------------------------
# COCO -> new-detector shape-safe weight transfer
# --------------------------------------------------------------------------------------------------
_TRANSFER_PREFIXES = ("rpn.", "roi_heads.box_head.", "roi_heads.mask_head.",
                      "roi_heads.mask_predictor.conv5_mask", "backbone.fpn.layer_blocks")
_NEVER_COPY = ("roi_heads.box_predictor", "roi_heads.mask_predictor.mask_fcn_logits", "backbone.body")


def transfer_coco_weights(model: nn.Module, log=print) -> dict:
    """Copy tensors from torchvision's COCO Mask R-CNN where name AND shape match.
    Never copies box classifier/regressor or the mask class predictor (115 PlantSeg classes)."""
    src = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).state_dict()
    dst = model.state_dict()
    copied, skipped_policy, skipped_mismatch = {}, [], []
    for k, v in src.items():
        if any(k.startswith(p) for p in _NEVER_COPY):
            skipped_policy.append(k)
        elif k in dst and dst[k].shape == v.shape and any(k.startswith(p) for p in _TRANSFER_PREFIXES):
            copied[k] = v
        else:
            skipped_mismatch.append(k)
    dst.update(copied)
    model.load_state_dict(dst)
    n_par = sum(v.numel() for v in copied.values())
    info = {"transferred_tensors": len(copied), "skipped_tensors": len(skipped_policy) + len(skipped_mismatch),
            "skipped_by_policy": len(skipped_policy), "skipped_name_or_shape_mismatch": len(skipped_mismatch),
            "transferred_parameters": n_par, "transferred_groups": sorted({".".join(k.split(".")[:3]) for k in copied})}
    log(f"[TRANSFER] transferred tensors     : {info['transferred_tensors']}")
    log(f"[TRANSFER] skipped tensors         : {info['skipped_tensors']} "
        f"(policy={info['skipped_by_policy']}, name/shape mismatch={info['skipped_name_or_shape_mismatch']})")
    log(f"[TRANSFER] transferred parameters  : {n_par} ({n_par / 1e6:.3f} M)")
    log(f"[TRANSFER] groups                  : {info['transferred_groups']}")
    return info


# --------------------------------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------------------------------
def create_tier1(num_classes=NUM_MASKRCNN_CLASSES):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                  box_score_thresh=EVAL_BOX_SCORE_THRESH,
                                  box_detections_per_img=EVAL_DETS_PER_IMG)
    inf = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inf, num_classes)
    infm = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(infm, 256, num_classes)
    model.transform = make_transform(yolo_style=False)
    return model


# --------------------------------------------------------------------------------------------------
# Tier 2 (+ light box head option used by ablation T3-B)
# --------------------------------------------------------------------------------------------------
class LightBoxHead(nn.Module):
    """h_i = ReLU(W vec(RoIAlign(F,B_i)) + b), dim 512."""

    def __init__(self, in_ch=256, roi=7, dim=512):
        super().__init__()
        self.fc = nn.Linear(in_ch * roi * roi, dim)
        self.dim = dim

    def forward(self, x):
        return F.relu(self.fc(x.flatten(1)), inplace=True)


def make_anchor_generator(use_p2: bool, anchors: str):
    if anchors == "derived":
        sizes = DERIVED_ANCHOR_SIZES_P2_P6 if use_p2 else None
        if sizes is None:   # P3..P6 train-derived ladder (analyze_anchor_distribution.py "sizes_P3_P6")
            sizes = (16, 40, 128, 400)
        ratios = DERIVED_ANCHOR_RATIOS
    else:
        sizes = DEFAULT_SIZES_P2_P6 if use_p2 else DEFAULT_SIZES_P3_P6
        ratios = DEFAULT_RATIOS
    return AnchorGenerator(sizes=tuple((s,) for s in sizes), aspect_ratios=(ratios,) * len(sizes))


def create_tier2(num_classes=NUM_MASKRCNN_CLASSES, use_p2=True, anchors="derived", transfer=False,
                 light_box_head=False, freeze_first=0, log=print):
    backbone = YOLO26BackboneFPN(use_p2=use_p2, freeze_first=freeze_first)
    names = ["0", "1", "2", "3"] if use_p2 else ["0", "1", "2"]
    model = MaskRCNN(
        backbone=backbone, num_classes=num_classes,
        rpn_anchor_generator=make_anchor_generator(use_p2, anchors),
        box_roi_pool=MultiScaleRoIAlign(names, output_size=7, sampling_ratio=2),
        mask_roi_pool=MultiScaleRoIAlign(names, output_size=14, sampling_ratio=2),
        min_size=MIN_SIZE, max_size=MAX_SIZE,
        box_score_thresh=EVAL_BOX_SCORE_THRESH, box_detections_per_img=EVAL_DETS_PER_IMG,
    )
    model.transform = make_transform(yolo_style=True)
    if light_box_head:
        model.roi_heads.box_head = LightBoxHead()
        model.roi_heads.box_predictor = FastRCNNPredictor(512, num_classes)
    model.pretraining_info = {"backbone": "YOLO26-seg pretrained", "yolo_pyramid": backbone.pyramid_info}
    if transfer:
        model.pretraining_info["coco_transfer"] = transfer_coco_weights(model, log=log)
    return model


# --------------------------------------------------------------------------------------------------
# Tier 3: RoI-aligned prototype mask head
# --------------------------------------------------------------------------------------------------
class RoIAlignedProtoMaskHead(nn.Module):
    """P = ProtoNet(F_P2) in R^{K x Hp x Wp};  P_i = RoIAlign(P, B_i) in R^{K x 28 x 28};
       alpha_i = g(h_i) in R^K;  Z_i(u,v) = sum_k alpha_ik P_ik(u,v);  M_i = sigmoid(Z_i)."""

    def __init__(self, channels=256, k=32, roi_size=28, coeff_hidden=256, rep_dim=512):
        super().__init__()
        self.k = k
        self.proto_net = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, k, 1),
        )
        self.proto_roi_align = MultiScaleRoIAlign(featmap_names=["proto"], output_size=roi_size, sampling_ratio=2)
        self.coeff_head = nn.Sequential(
            nn.Linear(rep_dim, coeff_hidden), nn.ReLU(inplace=True), nn.Linear(coeff_hidden, k))

    def forward(self, p_feature, roi_representation, proposals, image_shapes):
        proto = self.proto_net(p_feature)
        proto_roi = self.proto_roi_align(OrderedDict([("proto", proto)]), proposals, image_shapes)  # N,K,28,28
        coeff = self.coeff_head(roi_representation)                                                 # N,K
        logits = torch.einsum("nk,nkhw->nhw", coeff, proto_roi)
        return logits, proto, coeff


class ProtoRoIHeads(RoIHeads):
    """Torchvision RoIHeads with NO conventional mask branch: one 512-d RoI representation h_i feeds
    classification, box regression and prototype coefficients."""

    def __init__(self, box_roi_pool, box_head, box_predictor, mask_head: RoIAlignedProtoMaskHead,
                 proto_key="0", mask_loss_weight=1.0, **kw):
        super().__init__(box_roi_pool, box_head, box_predictor, **kw)
        self.proto_mask_head = mask_head
        self.proto_key = proto_key
        self.mask_loss_weight = mask_loss_weight
        self.mask_size = 28
        self.last_diag = {}

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        h = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(h)
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            loss_cls, loss_box = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            pos_masks = [lbl > 0 for lbl in labels]                   # positive foreground RoIs ONLY
            pos_props = [p[m] for p, m in zip(proposals, pos_masks)]
            pos_matched = [mi[m] for mi, m in zip(matched_idxs, pos_masks)]
            h_pos = h[torch.cat(pos_masks)]
            n_pos = int(h_pos.shape[0])
            if n_pos > 0:
                logits, _, _ = self.proto_mask_head(features[self.proto_key], h_pos, pos_props, image_shapes)
                gt_masks = [t["masks"] for t in targets]
                tgt = torch.cat([project_masks_on_boxes(gm, p, mi, self.mask_size)
                                 for gm, p, mi in zip(gt_masks, pos_props, pos_matched)], 0)
                loss_mask = F.binary_cross_entropy_with_logits(logits, tgt)
                with torch.no_grad():
                    self.last_diag = {"num_positive": n_pos, "logit_min": float(logits.min()),
                                      "logit_max": float(logits.max()), "logit_std": float(logits.std()),
                                      "pred_fg_ratio": float((logits > 0).float().mean()),
                                      "target_fg_ratio": float((tgt > 0.5).float().mean()),
                                      "shape_ok": tuple(logits.shape[-2:]) == (self.mask_size, self.mask_size)
                                                  and logits.shape == tgt.shape}
            else:
                loss_mask = sum(p.sum() for p in self.proto_mask_head.parameters()) * 0.0
                self.last_diag = {"num_positive": 0}
            losses = {"loss_classifier": loss_cls, "loss_box_reg": loss_box,
                      "loss_mask": self.mask_loss_weight * loss_mask}
            return result, losses

        boxes, scores, labels_ = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        n_per = [len(b) for b in boxes]
        if sum(n_per) == 0:
            masks = [torch.zeros((0, 1, self.mask_size, self.mask_size), device=features[self.proto_key].device)
                     for _ in boxes]
        else:    # mask RoIs are the FINAL boxes (as in Mask R-CNN), so re-pool h at the final boxes
            hf = self.box_head(self.box_roi_pool(features, boxes, image_shapes))
            logits, _, _ = self.proto_mask_head(features[self.proto_key], hf, boxes, image_shapes)
            masks = [m[:, None] for m in logits.sigmoid().split(n_per, 0)]
        for b, s, l, m in zip(boxes, scores, labels_, masks):
            result.append({"boxes": b, "labels": l, "scores": s, "masks": m})
        return result, losses


class ProtoMaskRCNN(GeneralizedRCNN):
    def __init__(self, num_classes=NUM_MASKRCNN_CLASSES, k=32, use_p2=True, anchors="derived", transfer=True,
                 mask_loss_weight=1.0, freeze_first=0, log=print):
        backbone = YOLO26BackboneFPN(use_p2=use_p2, freeze_first=freeze_first)
        rpn_anchors = make_anchor_generator(use_p2, anchors)
        rpn = RegionProposalNetwork(
            rpn_anchors, RPNHead(backbone.out_channels, rpn_anchors.num_anchors_per_location()[0]),
            0.7, 0.3, 256, 0.5, dict(training=2000, testing=1000), dict(training=2000, testing=1000), 0.7)
        names = ["0", "1", "2", "3"] if use_p2 else ["0", "1", "2"]
        roi_heads = ProtoRoIHeads(
            MultiScaleRoIAlign(names, output_size=7, sampling_ratio=2), LightBoxHead(),
            FastRCNNPredictor(512, num_classes),
            RoIAlignedProtoMaskHead(backbone.out_channels, k=k),
            proto_key="0", mask_loss_weight=mask_loss_weight,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=EVAL_BOX_SCORE_THRESH, nms_thresh=0.5,
            detections_per_img=EVAL_DETS_PER_IMG)
        super().__init__(backbone, rpn, roi_heads, make_transform(yolo_style=True))
        self.k = k
        self.pretraining_info = {"backbone": "YOLO26-seg pretrained", "yolo_pyramid": backbone.pyramid_info}
        if transfer:
            self.pretraining_info["coco_transfer"] = transfer_coco_weights(self, log=log)


# --------------------------------------------------------------------------------------------------
# Registry used by training / validation / benchmarking
# --------------------------------------------------------------------------------------------------
# recipe: "flat"   = V1-style single LR 3e-4 (first 4 YOLO layers frozen), cosine, warm-up
#         "staged" = discriminative LR groups + staged unfreezing (Part 13/14)
VARIANTS = {
    "t1":   dict(kind="t1", recipe="disc"),
    "t2_a": dict(kind="t2", use_p2=False, anchors="default", transfer=False, recipe="flat"),
    "t2_b": dict(kind="t2", use_p2=True, anchors="default", transfer=False, recipe="flat"),
    "t2_c": dict(kind="t2", use_p2=True, anchors="derived", transfer=False, recipe="flat"),
    "t2_d": dict(kind="t2", use_p2=True, anchors="derived", transfer=True, recipe="flat"),
    "t2_e": dict(kind="t2", use_p2=True, anchors="derived", transfer=True, recipe="staged"),
    # Tier 3 ablations are built on the BEST Tier-2 recipe (default t2_e, overridable after validation)
    "t3_a": dict(kind="t2", use_p2=True, anchors="derived", transfer=True, recipe="staged"),   # = best T2 + std mask head
    "t3_b": dict(kind="t2", use_p2=True, anchors="derived", transfer=True, recipe="staged", light_box_head=True),
    "t3_c": dict(kind="t3", k=32, recipe="staged"),
    "t3_k16": dict(kind="t3", k=16, recipe="staged"),
    "t3_k32": dict(kind="t3", k=32, recipe="staged"),
    "t3_k64": dict(kind="t3", k=64, recipe="staged"),
}


def create_variant(name: str, num_classes=NUM_MASKRCNN_CLASSES, log=print, **override):
    cfg = dict(VARIANTS[name]); cfg.update(override)
    kind = cfg["kind"]
    if kind == "t1":
        m = create_tier1(num_classes)
        m.pretraining_info = {"backbone": "COCO Mask R-CNN ResNet50-FPN (all components)"}
    elif kind == "t2":
        flat = cfg["recipe"] == "flat"
        m = create_tier2(num_classes, use_p2=cfg["use_p2"], anchors=cfg["anchors"], transfer=cfg["transfer"],
                         light_box_head=cfg.get("light_box_head", False), freeze_first=4 if flat else 0, log=log)
    elif kind == "t3":
        m = ProtoMaskRCNN(num_classes, k=cfg["k"], mask_loss_weight=cfg.get("mask_loss_weight", 1.0), log=log)
    else:
        raise ValueError(name)
    m.variant_name, m.variant_cfg = name, cfg
    return m


def describe_pretraining(model) -> dict:
    return getattr(model, "pretraining_info", {})


# backward-compatible helpers for older callers
def create_tier3(num_classes=NUM_MASKRCNN_CLASSES, k=32):
    return create_variant("t3_c", num_classes, k=k)


def create_model(tier):
    return create_variant({1: "t1", 2: "t2_e", 3: "t3_c"}[tier]) if isinstance(tier, int) else create_variant(tier)
