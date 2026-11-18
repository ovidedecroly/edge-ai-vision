"""
Master Orchestration Script for Mask R-CNN vs YOLO26-seg Research Experiment.
Executes data setup, model benchmarking, visualization, and manuscript generation.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_step(step_name: str, script_path: Path):
    print("\n" + "=" * 70)
    print(f">> RUNNING: {step_name}")
    print(f"   Script : {script_path}")
    print("=" * 70)
    cmd = [sys.executable, str(script_path)]
    subprocess.check_call(cmd, cwd=str(BASE_DIR))

def main():
    print("======================================================================")
    print("EXPERIMENT PIPELINE: MASK R-CNN vs YOLO26-SEG ON CARPARTS-SEG")
    print("Targeted for Q1 IEEE Transactions / Computer Vision Journals")
    print("======================================================================")
    
    # Step 1: Benchmark Latency & Parameters
    run_step("Architecture Benchmarks (GPU & CPU)", BASE_DIR / "scripts" / "benchmark_suite.py")
    
    # Step 2: Generate Qualitative Visual Figures
    run_step("Qualitative Comparison Figure Generation", BASE_DIR / "scripts" / "visual_comparison.py")
    
    # Step 3: Generate Paper Manuscripts (.docx & .md)
    run_step("Academic Paper Manuscript Generation", BASE_DIR / "scripts" / "generate_paper_doc.py")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Manuscript (Word .docx) : {BASE_DIR / 'paper' / 'manuscript_maskrcnn_yolo26.docx'}")
    print(f"Manuscript (Markdown)   : {BASE_DIR / 'paper' / 'manuscript_maskrcnn_yolo26.md'}")
    print(f"Comparison Figure       : {BASE_DIR / 'paper' / 'figures' / 'visual_comparison.png'}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
