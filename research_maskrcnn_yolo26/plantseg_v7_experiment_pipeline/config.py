from pathlib import Path
import torch

PROJECT_ROOT = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26")
DATASET_ROOT = PROJECT_ROOT / "plantseg"
YOLO26_WEIGHTS = Path("/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt")

OUTPUT_ROOT = PROJECT_ROOT / "runs" / "plantseg_v7_final"
PREPARED_ROOT = OUTPUT_ROOT / "_prepared_v7"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

NUM_DISEASE_CLASSES = 115
NUM_MASKRCNN_CLASSES = 116  # background + 115 disease classes
IMAGE_SIZE = 640
SEED = 42

EXPECTED_IMAGES = {"train": 5367, "val": 846, "test": 1561}
EXPECTED_VALID_ANN = {"train": 38251, "val": 6460, "test": 11103}

def resolve_dataset_dir(root: Path = DATASET_ROOT) -> Path:
    root = Path(root)
    candidates = [
        root,
        root / "plantseg",
    ]
    for c in candidates:
        if (c / "Metadata.csv").exists() and (c / "annotation_train.json").exists():
            return c
    raise FileNotFoundError(
        f"Could not locate extracted PlantSeg v7 under {root}. "
        f"Expected Metadata.csv and annotation_train.json."
    )

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# V2 experiment root (the V1 root `runs/plantseg_v7_final` is NEVER written to).
# ----------------------------------------------------------------------------
V2_ROOT = PROJECT_ROOT / "runs" / "plantseg_v7_v2"
V2_AUDITS = V2_ROOT / "audits"
V2_ANCHORS = V2_ROOT / "anchors"
V2_VALRES = V2_ROOT / "validation_results"
V2_BENCH = V2_ROOT / "benchmarks"
V2_DEBUG = V2_ROOT / "figures_debug"
# Mask R-CNN-family aspect-preserving resize (GeneralizedRCNNTransform)
MIN_SIZE = 640
MAX_SIZE = 1024
# Test split is blocked in V2 development. Set True ONLY after user approves a frozen model.
ALLOW_TEST_EVAL = False
