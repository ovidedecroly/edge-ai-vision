import sys
import subprocess
from pathlib import Path

def run_step(cmd, desc):
    print(f"\n{'='*70}\n[START] {desc}\nCMD: {' '.join(cmd)}\n{'='*70}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[SUCCESS] {desc}\n{'='*70}\n", flush=True)

def main():
    py = [sys.executable, "-u"]
    pipeline_dir = Path(__file__).resolve().parent
    results_dir = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/runs/plantseg_v7_final/results")

    # Tier 4 training is ALREADY COMPLETE (saved in tier4_yolo26seg-2)
    print("[INFO] Tier 4 training already finished successfully. Proceeding to evaluations.", flush=True)

    # 1. Evaluate all 4 Tiers on Test Set
    for t in [1, 2, 3, 4]:
        out_json = results_dir / f"tier{t}_test_metrics.json"
        if out_json.exists():
            print(f"[SKIP] Tier {t} evaluation already exists at {out_json}", flush=True)
            continue
        run_step(py + [str(pipeline_dir / "04_evaluate.py"), "--tier", str(t)],
                 f"Evaluating Tier {t} on Test Set")

    # 2. Efficiency Benchmark
    run_step(py + [str(pipeline_dir / "05_benchmark.py"), "--all"],
             "Benchmarking All 4 Tiers (FPS, Latency, GFLOPs, Params)")

    # 3. Generate Visual Comparison
    run_step(py + [str(pipeline_dir / "06_visualize.py"), "--num-images", "6"],
             "Generating Visual Comparison Figures")

    print("\n[ALL FINISHED] Full 4-Tier Evaluation and Benchmark Pipeline Completed Successfully!", flush=True)

if __name__ == "__main__":
    main()
