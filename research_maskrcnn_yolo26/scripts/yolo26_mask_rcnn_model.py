"""
YOLO26-Backboned Mask R-CNN Implementation
Hybrid Two-Stage Instance Segmentation Architecture:
Replaces standard ResNet-50 backbone with YOLO26 multi-scale feature extractor (C3, C4, C5),
coupled with Feature Pyramid Network (FPN) and standard Mask R-CNN RoI heads.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from ultralytics import YOLO


class YOLO26BackboneFPN(nn.Module):
    """
    Multi-Scale Feature Extractor using YOLO26 Backbone + torchvision Feature Pyramid Network (FPN).
    Extracts features from stages C3 (stride 8), C4 (stride 16), and C5 (stride 32).
    """
    def __init__(
        self,
        yolo_weights: str = "/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt",
        out_channels: int = 256,
        trainable_backbone_layers: int = 5
    ):
        super().__init__()
        # Load YOLO model
        yolo = YOLO(yolo_weights)
        # Backbone is defined by layers 0 to 10
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
        
        # Multi-scale FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[c3_ch, c4_ch, c5_ch],
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels
        
        # Freeze initial layers if specified
        if trainable_backbone_layers < 11:
            freeze_until = 11 - trainable_backbone_layers
            for i in range(freeze_until):
                for p in self.backbone_layers[i].parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        
        features = OrderedDict([("0", c3), ("1", c4), ("2", c5)])
        return self.fpn(features)


def create_yolo26_mask_rcnn(
    num_classes: int = 24,
    yolo_weights: str = "/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt",
    fpn_channels: int = 256,
    pretrained_backbone: bool = True
) -> MaskRCNN:
    """
    Constructs a complete two-stage Mask R-CNN model using the YOLO26-FPN backbone.
    num_classes includes the background class (e.g., 23 carparts classes + 1 background = 24).
    """
    backbone = YOLO26BackboneFPN(
        yolo_weights=yolo_weights,
        out_channels=fpn_channels,
        trainable_backbone_layers=7 if pretrained_backbone else 11
    )
    
    # Anchor generator matching 4 levels: 0 (stride 8), 1 (stride 16), 2 (stride 32), pool (stride 64)
    rpn_anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        min_size=640,
        max_size=640
    )
    return model


if __name__ == "__main__":
    model = create_yolo26_mask_rcnn(num_classes=24)
    model.eval()
    x = [torch.randn(3, 640, 640)]
    with torch.no_grad():
        preds = model(x)
    print("[SUCCESS] YOLO26-Backboned Mask R-CNN instantiated & verified successfully!")
    print("Boxes shape :", preds[0]["boxes"].shape)
    print("Masks shape :", preds[0]["masks"].shape)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parameters: {params:.2f} M")
