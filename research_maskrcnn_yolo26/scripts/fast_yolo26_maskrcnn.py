"""
Fast YOLO26-Mask R-CNN Hybrid Model
Combines:
1. YOLO26 backbone + Feature Pyramid Network (FPN)
2. Two-stage Region Proposal & MultiScaleRoIAlign
3. One-stage Prototype Mask Generation (ProtoNet) & Fast Matrix Assembly
"""

import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection import MaskRCNN
from ultralytics import YOLO


class FastYOLO26MaskRCNN(nn.Module):
    """
    Fast Two-Stage Instance Segmenter:
    - Backbone: YOLO26 stages C3, C4, C5 + FPN (256 channels)
    - ProtoNet: 32 prototype masks generated once globally from P3
    - Lightweight RoI Box Head: 1-FC head (predicts bounding box, class, and 32 mask coefficients)
    - Mask Assembly: Ultra-fast matrix multiplication (coefficients x prototype masks)
    """
    def __init__(
        self,
        num_classes: int = 116,
        yolo_weights: str = "/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt",
        num_prototypes: int = 32,
        fpn_channels: int = 256
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        
        # 1. YOLO26 Backbone
        yolo = YOLO(yolo_weights)
        self.backbone_layers = nn.ModuleList([yolo.model.model[i] for i in range(11)])
        
        # Probe channel dimensions dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 640, 640)
            out = dummy
            c3_ch, c4_ch, c5_ch = None, None, None
            for i, layer in enumerate(self.backbone_layers):
                out = layer(out)
                if i == 4:
                    c3_ch = out.shape[1]
                elif i == 6:
                    c4_ch = out.shape[1]
                elif i == 10:
                    c5_ch = out.shape[1]

        # 2. Multi-scale FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[c3_ch, c4_ch, c5_ch],
            out_channels=fpn_channels,
            extra_blocks=LastLevelMaxPool()
        )

        # 3. Global ProtoNet (runs once on P3 feature map)
        self.protonet = nn.Sequential(
            nn.Conv2d(fpn_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_prototypes, kernel_size=1)
        )

        # 4. RoIAlign
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2'],
            output_size=7,
            sampling_ratio=2
        )

        # 5. Lightweight Box Head + Mask Coefficient Head
        # Output: num_classes * 4 (boxes) + num_classes (class logits) + num_prototypes (coefficients)
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fpn_channels * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes * 4 + num_classes + num_prototypes)
        )

    def forward(self, x: torch.Tensor, proposals: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        out = x
        c3, c4, c5 = None, None, None
        for i, layer in enumerate(self.backbone_layers):
            out = layer(out)
            if i == 4:
                c3 = out
            elif i == 6:
                c4 = out
            elif i == 10:
                c5 = out

        features = self.fpn(OrderedDict([("0", c3), ("1", c4), ("2", c5)]))

        # Generate global prototypes once: [B, num_prototypes, H/8, W/8]
        proto = self.protonet(features["0"])

        # Default proposals for benchmarking if none passed
        if proposals is None:
            # 50 candidate proposal boxes across the image
            boxes = torch.tensor([
                [10.0, 10.0, 200.0, 200.0],
                [50.0, 50.0, 300.0, 300.0]
            ], device=x.device).repeat(25, 1) # 50 proposals
            proposals = [boxes for _ in range(B)]

        roi_feats = self.roi_align(features, proposals, [(H, W) for _ in range(B)])
        preds = self.box_head(roi_feats)
        
        # Mask coefficients are the last `num_prototypes` elements
        coeffs = preds[:, -self.num_prototypes:]
        
        # Fast assembly via matrix multiplication (BlendMask / YOLO style)
        # [num_rois, num_prototypes] @ [num_prototypes, H_proto * W_proto] -> [num_rois, H_proto * W_proto]
        proto_flat = proto[0].view(self.num_prototypes, -1)
        masks = torch.sigmoid(torch.matmul(coeffs, proto_flat))
        
        return preds, masks
