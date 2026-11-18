import torch
from torch.utils.data import DataLoader

from config import PREPARED_ROOT, get_device
from dataset_v7 import PlantSegCOCODataset, collate_fn
from models import create_model

def main():
    assert (PREPARED_ROOT / "coco" / "annotation_train.clean.json").exists(), \
        "Run 01_prepare_v7.py first"

    ds = PlantSegCOCODataset("train")
    print(f"Dataset samples: {len(ds)}")
    img, tgt = ds[0]
    print("Image:", img.shape)
    print("Boxes:", tgt["boxes"].shape, "Masks:", tgt["masks"].shape, "Labels:", tgt["labels"].shape)

    device = get_device()
    for tier in [1,2,3]:
        print(f"\nSmoke testing tier {tier}")
        model = create_model(tier).to(device)
        model.train()
        images = [img.to(device)]
        targets = [{
            k: v.to(device) for k,v in tgt.items()
            if torch.is_tensor(v) and k != "orig_size"
        }]
        losses = model(images, targets)
        print({k: float(v.detach().cpu()) for k,v in losses.items()})

        model.eval()
        with torch.no_grad():
            pred = model(images)[0]
        print("Prediction keys:", pred.keys())
        print("N detections:", len(pred["boxes"]))
    print("\nSMOKE TEST PASSED")

if __name__ == "__main__":
    main()
