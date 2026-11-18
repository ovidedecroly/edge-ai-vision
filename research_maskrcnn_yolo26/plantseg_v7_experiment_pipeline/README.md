# PlantSeg v7 — Cross-Paradigm Instance Segmentation Pipeline

This package is designed for the final PlantSeg v7 experiments discussed for the paper.

## Dataset location

Default dataset root:

`/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/plantseg`

The code auto-detects whether the extracted files are directly in that directory or in the nested directory:

`.../plantseg/plantseg/`

Expected v7 structure:

- `Metadata.csv`
- `annotation_train.json`
- `annotation_val.json`
- `annotation_test.json`
- `images/train`, `images/val`, `images/test`
- `annotations/train`, `annotations/val`, `annotations/test`

## Four models

1. Tier 1 — Mask R-CNN ResNet50-FPN
2. Tier 2 — YOLO26 backbone + FPN + standard Mask R-CNN heads
3. Tier 3 — Fast YOLO26-Mask R-CNN with:
   - YOLO26 backbone
   - FPN
   - RPN
   - proposal-specific Light Box Head
   - global ProtoNet
   - proposal-conditioned prototype coefficients
4. Tier 4 — YOLO26n-seg

## Ground-truth unit

One **manually delineated disease-symptom polygon region** in the official COCO JSON is treated as one symptom-region instance.

Connected-component reconstruction from semantic PNG masks is NOT used.

## Important v7 integrity handling

The official v7 COCO JSON may contain orphan annotations whose `image_id` is not present in the corresponding `images` array. The pipeline filters them automatically and writes sanitized JSON copies.

Expected image counts:

- train: 5367
- val: 846
- test: 1561
- total: 7774

Expected valid polygon counts after filtering (audit values):

- train: 38251
- val: 6460
- test: 11103
- total: 55814

The pipeline validates these but does not silently overwrite unexpected values.

## Setup

```bash
cd plantseg_v7_experiment_pipeline
pip install -r requirements.txt
```

Make sure the local YOLO26 segmentation checkpoint exists. Default:

`/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt`

## Step 1 — Audit and sanitize v7

```bash
python 01_prepare_v7.py
```

This will create:

- `_prepared_v7/coco/annotation_train.clean.json`
- `_prepared_v7/coco/annotation_val.clean.json`
- `_prepared_v7/coco/annotation_test.clean.json`
- `_prepared_v7/yolo/`
- `_prepared_v7/audit_report.json`

## Step 2 — Smoke-test models and dataset

```bash
python 02_smoke_test.py
```

Run this BEFORE a long training job.

## Step 3 — Train

Recommended controlled protocol:

```bash
python 03_train.py --tier 1 --epochs 50
python 03_train.py --tier 2 --epochs 50
python 03_train.py --tier 3 --epochs 50
python 03_train.py --tier 4 --epochs 50
```

Default controlled settings for Tiers 1–3:
- input 640
- AdamW
- lr 3e-4
- weight decay 1e-4
- cosine schedule
- same train/val split
- model selection by validation Mask AP, not cross-model loss

Tier 4 uses Ultralytics training because its loss/optimizer implementation is architecture-native.

## Step 4 — Final test evaluation

```bash
python 04_evaluate.py --tier 1
python 04_evaluate.py --tier 2
python 04_evaluate.py --tier 3
python 04_evaluate.py --tier 4
```

Outputs:
- Mask AP@[.50:.95]
- Mask AP50
- Mask AP75
- Mask APs/APm/APl
- Box AP@[.50:.95]
- Box AP50
- Box AP75
- AR100
- semantic mIoU
- semantic mAcc
- per-class IoU

The semantic metrics are calculated from actual semantic projections of instance predictions. They are NOT approximated from AP.

## Step 5 — Efficiency benchmark

```bash
python 05_benchmark.py --all
```

Default main benchmark:
- 640x640
- batch 1
- same device
- warmup 50
- timed iterations 200

## Step 6 — Real qualitative visualizations

```bash
python 06_visualize.py --num-images 6
```

This uses REAL model predictions and REAL COCO ground truth. No synthetic masks, jitter, or hard-coded scores are generated.

## Run everything

```bash
python run_all.py --epochs 50
```

Long training is intentionally not started automatically by the package creator. Run it in Antigravity after the smoke test passes.

## Scientific reporting notes

Use:
- `symptom-region instance segmentation`
- `manually delineated disease-symptom polygon region`

Avoid:
- `biologically verified individual lesion`
- pseudo-instance (when using official polygons directly)

Use instance AP as the primary metric and semantic mIoU/mAcc as secondary compatibility metrics with the original PlantSeg semantic benchmark.
