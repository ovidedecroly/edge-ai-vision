"""
run_v2_experiments.py: Master orchestration script for PlantSeg v7 V2 experiments.
Runs the required ablation experiments in strict sequence:

TIER 2:
1. T2-A: YOLO26 backbone + existing FPN (no P2, default anchors, no COCO transfer, flat LR)
2. T2-B: T2-A + P2 feature
3. T2-C: T2-B + train-derived anchors
4. T2-D: T2-C + compatible detector weight transfer
5. T2-E: T2-D + staged/discriminative learning rate

TIER 3:
6. T3-B: T2-E + Light Box Head (with standard mask head)
7. T3-C / T3-k32: T3-B + RoIAligned ProtoNet (k=32)
8. T3-k16: RoIAligned ProtoNet (k=16)
9. T3-k64: RoIAligned ProtoNet (k=64)

TIER 1 (Control):
10. T1: Mask R-CNN ResNet50-FPN on V2 clean pipeline
"""
import argparse
import sys
import subprocess
from pathlib import Path

from config import V2_ROOT
from models import VARIANTS


def run_experiment(variant, epochs=50, batch_size=4, val_every=2, workers=4):
    print("\n" + "=" * 80)
    print(f">> STARTING EXPERIMENT: {variant} (epochs={epochs}, bs={batch_size})")
    print("=" * 80, flush=True)
    
    cmd = [
        sys.executable, "03_train.py",
        "--variant", variant,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--val-every", str(val_every),
        "--workers", str(workers),
    ]
    subprocess.check_call(cmd)
    
    # Run compare script after completion
    try:
        subprocess.run([sys.executable, "compare_v2_validation.py"], check=False)
    except Exception as e:
        print(f"Warning: compare_v2_validation failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=None, help="List of models to run in sequence (e.g. --models t2_e t3_c)")
    ap.add_argument("--start-from", default=None, choices=list(VARIANTS.keys()))
    ap.add_argument("--only", default=None, choices=list(VARIANTS.keys()))
    args = ap.parse_args()

    default_order = [
        "t2_e",
        "t3_c",
        "t2_a",
        "t2_b",
        "t2_c",
        "t2_d",
        "t3_b",
        "t3_k16",
        "t3_k64",
        "t1"
    ]

    order = args.models or default_order

    if args.only:
        run_experiment(args.only, epochs=args.epochs, batch_size=args.batch_size,
                       val_every=args.val_every, workers=args.workers)
        return

    start_idx = 0
    if args.start_from and args.start_from in order:
        start_idx = order.index(args.start_from)

    for v in order[start_idx:]:
        run_experiment(v, epochs=args.epochs, batch_size=args.batch_size,
                       val_every=args.val_every, workers=args.workers)


if __name__ == "__main__":
    main()
