"""
Benchmarking & Evaluation Engine for Indonesian License Plate Detection.
Measures inference latency, FPS, memory footprint, and exports comparison tables for academic papers.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from ultralytics import YOLO


class ANPRBenchmarker:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))

    def benchmark_latency(
        self,
        imgsz: int = 640,
        warmup_runs: int = 20,
        test_runs: int = 100,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Measures precise per-frame inference latency (ms) and throughput (FPS)."""
        dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

        print(f"[BENCHMARK] Warming up model on {device} ({warmup_runs} iterations)...")
        for _ in range(warmup_runs):
            _ = self.model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)

        print(f"[BENCHMARK] Measuring {test_runs} iterations at {imgsz}x{imgsz}...")
        timings = []
        for _ in range(test_runs):
            t0 = time.perf_counter()
            _ = self.model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)
            dt = (time.perf_counter() - t0) * 1000.0
            timings.append(dt)

        avg_lat = float(np.mean(timings))
        std_lat = float(np.std(timings))
        p95_lat = float(np.percentile(timings, 95))
        fps = float(1000.0 / avg_lat) if avg_lat > 0 else 0.0

        res = {
            "device": device,
            "imgsz": imgsz,
            "latency_mean_ms": round(avg_lat, 2),
            "latency_std_ms": round(std_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
            "throughput_fps": round(fps, 1),
        }
        print(f" -> Latency: {res['latency_mean_ms']} ± {res['latency_std_ms']} ms | FPS: {res['throughput_fps']} FPS")
        return res

    def inspect_model_complexity(self) -> Dict[str, Any]:
        """Calculates parameters count, model file size, and architecture details."""
        size_mb = round(self.model_path.stat().st_size / (1024 * 1024), 2)
        params_count = sum(p.numel() for p in self.model.model.parameters())
        params_m = round(params_count / 1e6, 2)

        return {
            "model_name": self.model_path.name,
            "file_size_mb": size_mb,
            "params_millions": params_m,
        }

    @staticmethod
    def generate_latex_table(benchmark_results: List[Dict[str, Any]]) -> str:
        """Exports formatted LaTeX table for IEEE / JCCE journal submissions."""
        header = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Inference Efficiency and Latency Benchmarks on Edge Hardware}\n"
            "\\label{tab:latency_benchmarks}\n"
            "\\begin{tabular}{lcccccc}\n"
            "\\hline\n"
            "\\textbf{Model} & \\textbf{Device} & \\textbf{Input Size} & \\textbf{Params (M)} & \\textbf{Size (MB)} & \\textbf{Latency (ms)} & \\textbf{Throughput (FPS)} \\\\\n"
            "\\hline\n"
        )
        rows = []
        for r in benchmark_results:
            rows.append(
                f"{r['model_name']} & {r['device'].upper()} & {r['imgsz']}$\\times${r['imgsz']} & "
                f"{r['params_millions']} & {r['file_size_mb']} & "
                f"{r['latency_mean_ms']} $\\pm$ {r['latency_std_ms']} & \\textbf{{{r['throughput_fps']}}} \\\\"
            )
        footer = "\\hline\n\\end{tabular}\n\\end{table}\n"
        return header + "\n".join(rows) + "\n" + footer


if __name__ == "__main__":
    model_p = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/yolo26n.pt")
    if model_p.exists():
        bench = ANPRBenchmarker(model_p)
        complexity = bench.inspect_model_complexity()

        results = []
        for dev in ["cpu", "0"] if torch.cuda.is_available() else ["cpu"]:
            for sz in [320, 640]:
                lat = bench.benchmark_latency(imgsz=sz, device=dev)
                row = {**complexity, **lat}
                results.append(row)

        print("\n" + "=" * 60)
        print("LATEX BENCHMARK TABLE FOR PAPER:")
        print("=" * 60)
        print(bench.generate_latex_table(results))
