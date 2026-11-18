"""
Comprehensive Benchmarking Suite for:
1. Standard Mask R-CNN (ResNet-50-FPN)
2. YOLO26-Backboned Mask R-CNN (Proposed Hybrid)
3. Pure YOLO26-seg (One-Stage SOTA)
Measures latency, throughput (FPS), parameters, model size, and generates paper tables.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from ultralytics import YOLO

from yolo26_mask_rcnn_model import create_yolo26_mask_rcnn
from fast_yolo26_maskrcnn import FastYOLO26MaskRCNN


def benchmark_latency_and_fps(model, device, input_size=(1, 3, 640, 640), runs=30, warmup=10):
    model = model.to(device).eval()
    if isinstance(model, MaskRCNN):
        dummy_tensor = [torch.randn(3, input_size[2], input_size[3], device=device)]
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                _ = model(dummy_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0 / runs
    else:
        dummy_tensor = torch.randn(*input_size, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                _ = model(dummy_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0 / runs
    fps = 1000.0 / dt
    return dt, fps


class ArchitectureBenchmark:
    def __init__(self, yolo_seg_path: str = "/media/ovide/DATA/DINO_YOLO/yolo26n-seg.pt", num_classes: int = 24):
        self.num_classes = num_classes
        self.yolo_seg_path = yolo_seg_path
        
        # Instantiate models
        print("[INIT] Loading Standard Mask R-CNN (ResNet-50-FPN)...")
        self.std_maskrcnn = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        
        print("[INIT] Loading YOLO26-Backboned Mask R-CNN (Proposed)...")
        self.yolo_maskrcnn = create_yolo26_mask_rcnn(num_classes=num_classes, yolo_weights=yolo_seg_path)
        
        print("[INIT] Loading Pure YOLO26-seg...")
        self.pure_yolo = YOLO(yolo_seg_path)

    def count_parameters(self) -> Dict[str, float]:
        p_std = sum(p.numel() for p in self.std_maskrcnn.parameters()) / 1e6
        p_yolo_mask = sum(p.numel() for p in self.yolo_maskrcnn.parameters()) / 1e6
        p_pure = sum(p.numel() for p in self.pure_yolo.model.parameters()) / 1e6
        return {
            "Standard Mask R-CNN (ResNet-50)": round(p_std, 2),
            "YOLO26-Backboned Mask R-CNN": round(p_yolo_mask, 2),
            "Pure YOLO26-seg": round(p_pure, 2)
        }

    def measure_latency_gpu(self, imgsz: int = 640, warmup: int = 10, iters: int = 40) -> Dict[str, Dict[str, float]]:
        if not torch.cuda.is_available():
            return {}
        
        print(f"\n[BENCHMARK] Running GPU Benchmark on {torch.cuda.get_device_name(0)} (imgsz={imgsz})...")
        m1 = self.std_maskrcnn.cuda().eval()
        m2 = self.yolo_maskrcnn.cuda().eval()
        
        dummy_tensor = [torch.randn(3, imgsz, imgsz, device="cuda")]
        dummy_np = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        
        results = {}
        
        # 1. Standard Mask R-CNN
        for _ in range(warmup): _ = m1(dummy_tensor)
        torch.cuda.synchronize()
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = m1(dummy_tensor)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
        avg = float(np.mean(timings))
        results["Standard Mask R-CNN"] = {"latency_ms": round(avg, 2), "std_ms": round(float(np.std(timings)), 2), "fps": round(1000.0 / avg, 1)}

        # 2. YOLO26-Backboned Mask R-CNN
        for _ in range(warmup): _ = m2(dummy_tensor)
        torch.cuda.synchronize()
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = m2(dummy_tensor)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
        avg = float(np.mean(timings))
        results["YOLO26-Backboned Mask R-CNN"] = {"latency_ms": round(avg, 2), "std_ms": round(float(np.std(timings)), 2), "fps": round(1000.0 / avg, 1)}

        # 3. Pure YOLO26-seg
        for _ in range(warmup): _ = self.pure_yolo.predict(dummy_np, imgsz=imgsz, device="cuda", verbose=False)
        torch.cuda.synchronize()
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = self.pure_yolo.predict(dummy_np, imgsz=imgsz, device="cuda", verbose=False)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
        avg = float(np.mean(timings))
        results["Pure YOLO26-seg"] = {"latency_ms": round(avg, 2), "std_ms": round(float(np.std(timings)), 2), "fps": round(1000.0 / avg, 1)}
        
        return results

    def measure_latency_cpu(self, imgsz: int = 320, warmup: int = 2, iters: int = 10) -> Dict[str, Dict[str, float]]:
        print(f"\n[BENCHMARK] Running CPU Benchmark (imgsz={imgsz})...")
        m1 = self.std_maskrcnn.cpu().eval()
        m2 = self.yolo_maskrcnn.cpu().eval()
        
        dummy_tensor = [torch.randn(3, imgsz, imgsz)]
        dummy_np = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        
        results = {}
        
        # 1. Standard Mask R-CNN
        for _ in range(warmup): _ = m1(dummy_tensor)
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = m1(dummy_tensor)
            timings.append((time.perf_counter() - t0) * 1000.0)
        avg = float(np.mean(timings))
        results["Standard Mask R-CNN"] = {"latency_ms": round(avg, 2), "std_ms": round(float(np.std(timings)), 2), "fps": round(1000.0 / avg, 1)}

        # 2. YOLO26-Backboned Mask R-CNN
        for _ in range(warmup): _ = m2(dummy_tensor)
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = m2(dummy_tensor)
            timings.append((time.perf_counter() - t0) * 1000.0)
        avg = float(np.mean(timings))
        results["YOLO26-Backboned Mask R-CNN"] = {"latency_ms": round(avg, 2), "std_ms": round(float(np.std(timings)), 2), "fps": round(1000.0 / avg, 1)}

        # 3. Pure YOLO26-seg
        for _ in range(warmup): _ = self.pure_yolo.predict(dummy_np, imgsz=imgsz, device="cpu", verbose=False)
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = self.pure_yolo.predict(dummy_np, imgsz=imgsz, device="cpu", verbose=False)
            timings.append((time.perf_counter() - t0) * 1000.0)
        avg = float(np.mean(timings))
        results["Pure YOLO26-seg"] = {"latency_ms": round(avg, 2), "std_ms": round(float(np.std(timings)), 2), "fps": round(1000.0 / avg, 1)}
        
        return results


if __name__ == "__main__":
    bench = ArchitectureBenchmark()
    params = bench.count_parameters()
    print("\n================== PARAMETER COUNTS ==================")
    for k, v in params.items():
        print(f" * {k:35}: {v:.2f} M params")
    
    gpu_res = bench.measure_latency_gpu(imgsz=640)
    print("\n================== GPU BENCHMARK (640x640) ==================")
    for k, v in gpu_res.items():
        print(f" * {k:35}: {v['latency_ms']:.2f} ± {v['std_ms']:.2f} ms | {v['fps']:.1f} FPS")
    
    cpu_res = bench.measure_latency_cpu(imgsz=320)
    print("\n================== CPU BENCHMARK (320x320) ==================")
    for k, v in cpu_res.items():
        print(f" * {k:35}: {v['latency_ms']:.2f} ± {v['std_ms']:.2f} ms | {v['fps']:.1f} FPS")
    
    out_json = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/runs/benchmark_summary.json")
    with open(out_json, "w") as f:
        json.dump({"params": params, "gpu_640": gpu_res, "cpu_320": cpu_res}, f, indent=2)
    print(f"\n[INFO] Benchmark saved to: {out_json}")
