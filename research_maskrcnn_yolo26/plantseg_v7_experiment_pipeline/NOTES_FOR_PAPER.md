# Notes for manuscript reporting

## Dataset formulation
PlantSeg v7 is treated as a multi-representation dataset. The original publication benchmarked semantic segmentation, while the released COCO polygon objects preserve annotation-level identities for manually delineated disease-symptom regions.

## Instance unit
One COCO polygon annotation object = one symptom-region instance.

Do not claim the target represents a biologically validated individual lesion.

## Main metrics
Primary:
- Mask AP@[.50:.95]
- Mask AP50
- Mask AP75
- Mask APs/APm/APl
- Box AP@[.50:.95]
- Box AP50
- Box AP75

Secondary compatibility metrics:
- mIoU
- mAcc
- per-class IoU

## Split
Use only official v7 split:
- 5367 train
- 846 validation
- 1561 test

## Annotation filtering
Report neutrally:
"Annotations whose image identifiers were not represented in the corresponding v7 image index were excluded during preprocessing."

Do not label the official release as "buggy" in the paper.

## Model selection
Select checkpoints by validation Mask AP rather than comparing absolute loss values across architectures.

## Pareto plot
Use:
- x-axis: latency or FPS
- y-axis: Mask AP@[.50:.95]
- bubble size: parameters

Do NOT use cross-architecture validation loss on the y-axis.
