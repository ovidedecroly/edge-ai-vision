import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from config import V2_ROOT, YOLO26_WEIGHTS, PREPARED_ROOT
from models import VARIANTS
from train_utils import train_variant


def train_yolo(epochs=50, batch=8, name="tier4_yolo26seg_v2"):
    """Tier 4 reference. NOT part of the V2 development loop (V1 Tier-4 weights are reused); kept for completeness.
    Writes under runs/plantseg_v7_v2/ only (never into the V1 directory)."""
    yaml_path = PREPARED_ROOT / "yolo" / "plantseg_v7.yaml"
    yolo_bin = shutil.which("yolo") or str(Path(sys.executable).parent / "yolo")
    cmd = [yolo_bin, "segment", "train", f"model={YOLO26_WEIGHTS}", f"data={yaml_path}", f"epochs={epochs}",
           f"batch={batch}", "imgsz=640", "device=0", f"project={V2_ROOT / 'tier4'}", f"name={name}",
           "seed=42", "deterministic=True", "plots=True"]
    print(" ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description="V2 training (TRAIN + VALIDATION only).")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS) + ["tier4"])
    ap.add_argument("--epochs", type=int, default=50, help="50 epochs to match Mask R-CNN and YOLO26 baselines")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--mask-loss-weight", type=float, default=None, help="lambda_m (Tier 3); tune on VALIDATION only")
    ap.add_argument("--rfs-t", type=float, default=None, help="repeat-factor threshold; use only if val improves")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-iters", type=int, default=None, help="debug only")
    args = ap.parse_args()
    if args.variant == "tier4":
        train_yolo(args.epochs, max(1, args.batch_size * 2))
        return
    train_variant(args.variant, epochs=args.epochs, batch_size=args.batch_size, out_root=args.out,
                  val_every=args.val_every, mask_loss_weight=args.mask_loss_weight, rfs_t=args.rfs_t,
                  num_workers=args.workers, resume=args.resume, max_train_iters=args.max_train_iters)


if __name__ == "__main__":
    main()
