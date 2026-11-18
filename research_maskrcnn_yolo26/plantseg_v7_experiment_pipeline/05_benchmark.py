"""
05_benchmark.py (V2) -- identical protocol for every model:
  input 640x640 (real VALIDATION images, aspect-preserved letterbox with zero padding), batch 1, FP32,
  50 warm-up + 200 measured iterations, CUDA-synchronised timing, same GPU.
  FPS = 1000 / mean latency(ms).
Tier 3 runs its REAL inference path (RPN -> proposals -> NMS -> RoIAlign -> ProtoNet -> mask assembly);
no dummy proposals.  FLOPs: torch.utils.flop_counter.FlopCounterMode on the backbone+heads forward
(conv / linear / matmul only; RoIAlign, NMS, mask pasting are NOT counted -> GFLOPs are a lower bound
for the proposal-based models; same counter for every model).
`--random-init` benchmarks the (untrained) architecture; detections per image then do not reflect a trained model.
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from config import V2_ROOT, V2_BENCH, OUTPUT_ROOT, get_device, resolve_dataset_dir
from dataset_v7 import load_geomfix_coco
from models import create_variant, VARIANTS, count_params
from utils import torch_checkpoint_state, ensure_dirs

SIZE = 640


def letterbox_images(n=20, seed=0):
    coco = load_geomfix_coco("val")
    ds = resolve_dataset_dir()
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(coco["images"]), n, replace=False)
    out = []
    for i in sel:
        im = cv2.imread(str(ds / "images" / "val" / coco["images"][i]["file_name"]))   # EXIF applied
        h, w = im.shape[:2]
        s = SIZE / max(h, w)
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        out.append(canvas)
    return out


def timing(fn, inputs, warmup, iters, device):
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        x = inputs[i % len(inputs)]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"latency_ms_mean": float(a.mean()), "latency_ms_std": float(a.std()), "latency_ms_median": float(np.median(a)),
            "latency_ms_p95": float(np.percentile(a, 95)), "fps": float(1000.0 / a.mean())}


def gflops_torchvision(model, x):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model([x])
    return fc.get_total_flops() / 1e9


def bench_torchvision(variant, warmup=50, iters=200, device=None, random_init=False, ckpt=None):
    device = device or get_device()
    model = create_variant(variant, log=lambda *a: None).to(device)
    if not random_init:
        model.load_state_dict(torch_checkpoint_state(ckpt or V2_ROOT / variant / "best.pt", device), strict=True)
    model.eval()
    imgs = [torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)
            for a in letterbox_images()]
    ndet = []

    @torch.no_grad()
    def run(x):
        o = model([x])[0]
        ndet.append(len(o["scores"]))
        return o

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    r = timing(run, imgs, warmup, iters, device)
    r.update({"variant": variant, "params_M": count_params(model) / 1e6, "gflops_conv_linear_matmul": gflops_torchvision(model, imgs[0]),
              "mean_detections_per_image": float(np.mean(ndet[warmup:])), "random_init": random_init,
              "peak_vram_MB": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None})
    return r


def bench_yolo(weights, warmup=50, iters=200, device=None):
    from ultralytics import YOLO
    device = device or get_device()
    y = YOLO(str(weights))
    arrs = letterbox_images()
    dev = 0 if device.type == "cuda" else "cpu"
    ndet = []

    def run(a):
        r = y.predict(a, imgsz=SIZE, device=dev, conf=0.001, iou=0.7, max_det=100, verbose=False)[0]
        ndet.append(0 if r.boxes is None else len(r.boxes))

    r = timing(run, arrs, warmup, iters, device)
    net = y.model.to(device).eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        net(torch.zeros(1, 3, SIZE, SIZE, device=device))
    r.update({"variant": "tier4", "params_M": count_params(net) / 1e6, "gflops_conv_linear_matmul": fc.get_total_flops() / 1e9,
              "mean_detections_per_image": float(np.mean(ndet[warmup:])), "random_init": False,
              "note": "latency = Ultralytics predict() end-to-end incl. CPU letterbox pre-processing"})
    return r


def find_yolo_best():
    c = list((OUTPUT_ROOT / "yolo_runs").rglob("weights/best.pt"))   # V1 weights, read-only
    return max(c, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True, help="e.g. t1 t2_e t3_c tier4")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ensure_dirs(V2_BENCH)
    res = {}
    for v in a.variants:
        print("benchmark", v, flush=True)
        res[v] = bench_yolo(find_yolo_best(), a.warmup, a.iters) if v == "tier4" else \
            bench_torchvision(v, a.warmup, a.iters, random_init=a.random_init)
        print(json.dumps(res[v], indent=2), flush=True)
    p = V2_BENCH / f"efficiency_640_b1{a.tag}.json"
    p.write_text(json.dumps(res, indent=2))
    print("Saved", p)


if __name__ == "__main__":
    main()
