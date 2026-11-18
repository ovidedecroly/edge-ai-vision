"""
compare_v2_validation.py: Compare candidate V2 models ONLY on the VALIDATION set.
Reads validation metrics from runs/plantseg_v7_v2/{variant}/best.pt or validation_results/
and benchmark results from runs/plantseg_v7_v2/benchmarks/

Output table format:
| Model | Mask AP | AP50 | AP75 | APs | APm | APl | Box AP | Params | FPS |
"""
import json
from pathlib import Path
import torch
import pandas as pd
from tabulate import tabulate

from config import V2_ROOT, V2_VALRES, V2_BENCH, get_device
from models import create_variant, count_params, VARIANTS
from validation import evaluate_torchvision


def get_model_stats(variant_name):
    var_dir = V2_ROOT / variant_name
    best_pt = var_dir / "best.pt"
    history_json = var_dir / "history.json"
    bench_file = V2_BENCH / "efficiency_640_b1.json"
    
    val_metrics = {}
    best_epoch = None
    
    if best_pt.exists():
        try:
            ck = torch.load(best_pt, map_location="cpu", weights_only=False)
            best_epoch = ck.get("epoch", None)
            val_mask_ap = ck.get("val_Mask_AP", None)
        except Exception:
            pass
            
    if history_json.exists():
        try:
            history = json.loads(history_json.read_text())
            if best_epoch is not None:
                for h in history:
                    if h.get("epoch") == best_epoch and "val_metrics" in h:
                        val_metrics = h["val_metrics"]
                        break
            if not val_metrics and history:
                # find entry with highest val_Mask_AP
                best_entry = max(history, key=lambda x: x.get("val_Mask_AP", -1))
                val_metrics = best_entry.get("val_metrics", {})
                best_epoch = best_entry.get("epoch")
        except Exception:
            pass

    # check validation_results
    val_json = V2_VALRES / f"{variant_name}_val_metrics.json"
    if val_json.exists():
        try:
            val_metrics = json.loads(val_json.read_text())
        except Exception:
            pass

    # get benchmark stats if available
    fps = None
    params_m = None
    if bench_file.exists():
        try:
            bdata = json.loads(bench_file.read_text())
            if variant_name in bdata:
                fps = bdata[variant_name].get("fps", None)
                params_m = bdata[variant_name].get("params_M", None)
        except Exception:
            pass

    if params_m is None and variant_name in VARIANTS:
        try:
            m = create_variant(variant_name, log=lambda *a: None)
            params_m = count_params(m) / 1e6
            del m
        except Exception:
            pass

    return {
        "variant": variant_name,
        "best_epoch": best_epoch,
        "Mask_AP": val_metrics.get("Mask_AP", None),
        "AP50": val_metrics.get("Mask_AP50", None),
        "AP75": val_metrics.get("Mask_AP75", None),
        "APs": val_metrics.get("Mask_APs", None),
        "APm": val_metrics.get("Mask_APm", None),
        "APl": val_metrics.get("Mask_APl", None),
        "Box_AP": val_metrics.get("Box_AP", None),
        "Params_M": params_m,
        "FPS": fps,
    }


def main():
    variants_to_check = [
        "t1",
        "t2_a", "t2_b", "t2_c", "t2_d", "t2_e",
        "t3_a", "t3_b", "t3_c", "t3_k16", "t3_k32", "t3_k64",
        "tier4"
    ]
    
    rows = []
    for v in variants_to_check:
        stats = get_model_stats(v)
        if stats["Mask_AP"] is not None or (V2_ROOT / v).exists():
            rows.append(stats)
            
    if not rows:
        print("No trained V2 variants found yet under runs/plantseg_v7_v2/.")
        return

    headers = ["Model", "Epoch", "Mask AP", "AP50", "AP75", "APs", "APm", "APl", "Box AP", "Params (M)", "FPS"]
    table_data = []
    for r in rows:
        fmt = lambda val, d=2: f"{val:.{d}f}" if val is not None else "-"
        table_data.append([
            r["variant"],
            str(r["best_epoch"]) if r["best_epoch"] is not None else "-",
            fmt(r["Mask_AP"]),
            fmt(r["AP50"]),
            fmt(r["AP75"]),
            fmt(r["APs"]),
            fmt(r["APm"]),
            fmt(r["APl"]),
            fmt(r["Box_AP"]),
            fmt(r["Params_M"], 2),
            fmt(r["FPS"], 1) if r["FPS"] is not None else "-",
        ])

    print("\n" + "=" * 80)
    print("V2 CANDIDATE MODELS COMPARISON (VALIDATION SET ONLY)")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 80 + "\n")

    # save as csv and json
    df = pd.DataFrame(rows)
    csv_path = V2_VALRES / "compare_v2_validation.csv"
    json_path = V2_VALRES / "compare_v2_validation.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved summary to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
