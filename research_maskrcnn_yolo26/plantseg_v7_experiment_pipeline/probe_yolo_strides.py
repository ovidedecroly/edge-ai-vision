"""Dynamic probe of the YOLO26 backbone graph: print every layer's output shape/stride (no hard-coded indices)."""
import json, torch
from ultralytics import YOLO
from config import YOLO26_WEIGHTS, V2_AUDITS

def probe(weights=str(YOLO26_WEIGHTS), size=640):
    y = YOLO(weights)
    rows = []
    x = torch.zeros(1, 3, size, size)
    y.model.eval()
    with torch.no_grad():
        for i, layer in enumerate(y.model.model):
            if isinstance(getattr(layer, "f", -1), list) or type(layer).__name__ == "Concat" or (i > 0 and type(layer).__name__ == "Upsample"):
                break
            x = layer(x)
            if isinstance(x, (list, tuple)):
                break
            rows.append({"idx": i, "type": type(layer).__name__, "shape": list(x.shape),
                         "stride": size // x.shape[-1], "channels": x.shape[1]})
            print(i, type(layer).__name__, tuple(x.shape), "stride", size // x.shape[-1], flush=True)
    return rows

def select_pyramid(rows):
    """Last layer index at each stride (4/8/16/32) within the backbone = the deepest, most-refined feature at that scale."""
    sel = {}
    for r in rows:
        if r["stride"] in (4, 8, 16, 32):
            sel[r["stride"]] = r  # keep last one seen for the stride
    return sel

if __name__ == "__main__":
    rows = probe()
    sel = select_pyramid(rows)
    out = {"layers": rows, "selected_last_at_stride": {f"C{int(__import__('math').log2(s))}": v for s, v in sel.items()}}
    V2_AUDITS.mkdir(parents=True, exist_ok=True)
    (V2_AUDITS / "yolo26_stride_map.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["selected_last_at_stride"], indent=2))
