# Prompt for Antigravity

Use this package as the new clean PlantSeg v7 experiment pipeline.

Dataset root:
`/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/plantseg`

Do not reuse v2 checkpoints, old hard-coded metrics, synthetic qualitative figures, or old Pareto plots.

Tasks:

1. Create a fresh Python environment or use the project's current environment and install `requirements.txt`.
2. Run:
   `python 01_prepare_v7.py`
3. Verify that the audit reports:
   - 5367 train images
   - 846 validation images
   - 1561 test images
   - 38251 / 6460 / 11103 valid polygon annotations after image-ID filtering.
4. Run:
   `python 02_smoke_test.py`
5. If Tier 3 fails because of a torchvision API-version difference, patch ONLY the API compatibility issue. Do not replace the ProtoNet formulation with the previous conventional lightweight Mask R-CNN mask head.
6. Confirm Tier 3 training forward returns:
   - loss_objectness
   - loss_rpn_box_reg
   - loss_classifier
   - loss_box_reg
   - loss_mask
7. Confirm Tier 3 inference returns:
   - boxes
   - labels
   - scores
   - masks
8. Train all four tiers from scratch on the official v7 split using the new output directory.
9. For Tiers 1–3 use the same controlled optimizer/scheduler/epochs and select best checkpoint by validation Mask AP.
10. Evaluate all four on the untouched 1561-image test split.
11. Do not calculate mIoU from AP. Use the semantic-projection implementation in `04_evaluate.py`.
12. Generate new efficiency results from trained checkpoints.
13. Generate only real prediction figures; do not use synthetic masks, random jitter, or hard-coded scores.
14. Replace the old speed-vs-loss Pareto plot with speed-vs-Mask-AP.
15. Preserve all raw result JSON/CSV files and report exact commands, package versions, GPU, CPU, CUDA, PyTorch, torchvision, and Ultralytics versions.

Before long training, show me the output of `01_prepare_v7.py` and `02_smoke_test.py`.
