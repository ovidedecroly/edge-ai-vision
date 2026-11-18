import json
import random
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def sanitize_coco(coco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    valid_img_ids = {img["id"] for img in coco["images"]}
    valid_anns = [a for a in coco["annotations"] if a["image_id"] in valid_img_ids]
    orphan_anns = [a for a in coco["annotations"] if a["image_id"] not in valid_img_ids]
    out = dict(coco)
    out["annotations"] = valid_anns
    stats = {
        "images": len(coco["images"]),
        "annotations_total": len(coco["annotations"]),
        "annotations_valid": len(valid_anns),
        "annotations_orphan": len(orphan_anns),
        "categories": len(coco.get("categories", [])),
    }
    return out, stats

def category_map_from_coco(coco: Dict[str, Any]):
    cats = coco.get("categories", [])
    return {int(c["id"]): c.get("name", str(c["id"])) for c in cats}

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def torch_checkpoint_state(path: Path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj
