#!/usr/bin/env python3
"""
Master ANPR Research & Deployment Pipeline Runner.
Orchestrates Dataset Curation -> Training/Ablation -> Benchmarking -> Platform Deployment -> Paper Generation.
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="Master ANPR Research & Deployment Pipeline")
    parser.add_argument("--step", choices=["all", "curate", "train", "benchmark", "deploy", "paper"], default="all",
                        help="Specify pipeline step to execute")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu, 0, auto)")
    args = parser.parse_args()

    print("=" * 65)
    print("INDONESIAN ANPR RESEARCH & EDGE-AI DEPLOYMENT PIPELINE")
    print("=" * 65)

    if args.step in ["all", "curate"]:
        print("\n>>> [STEP 1/5] Executing Multi-Source Dataset Curation & Harmonization...")
        from scripts.dataset_curator import DatasetCurator
        out_data = BASE_DIR / "data" / "indonesian_plates_combined"
        curator = DatasetCurator(output_root=out_data)
        curator.initialize_dirs()
        # Curate local raw data
        local_raw = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/datasets/raw")
        if local_raw.exists():
            for sub in local_raw.glob("*/*"):
                if sub.is_dir() and any(sub.glob("*.txt")):
                    curator.process_source_folder("surakarta_cctv", sub, sub)
        curator.generate_yaml()
        curator.print_summary()

    if args.step in ["all", "train"]:
        print("\n>>> [STEP 2/5] Executing Systematic Training & Ablation Study...")
        from scripts.train_and_ablation import ANPRTrainer
        runs_dir = BASE_DIR / "data" / "runs"
        yaml_p = BASE_DIR / "data" / "indonesian_plates_combined" / "data.yaml"
        if yaml_p.exists():
            import torch
            dev = "0" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
            trainer = ANPRTrainer(runs_dir=runs_dir, data_yaml=yaml_p)
            trainer.run_training_experiment(
                experiment_id="exp_anpr_multisource_yolo26n",
                base_model="/media/ovide/DATA/PIT-UNS/edge-ai-vision/yolo26n.pt",
                epochs=args.epochs,
                device=dev,
                description="YOLO26n model trained on aggregated Indonesian license plate multi-source dataset."
            )
        else:
            print("[INFO] No data.yaml found; skipping training step (run curate first).")

    if args.step in ["all", "benchmark"]:
        print("\n>>> [STEP 3/5] Executing Inference Efficiency & Latency Benchmarks...")
        from scripts.benchmark_eval import ANPRBenchmarker
        best_p = BASE_DIR / "data" / "runs" / "exp_anpr_multisource_yolo26n" / "weights" / "best.pt"
        if not best_p.exists():
            best_p = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/yolo26n.pt")
        bench = ANPRBenchmarker(best_p)
        print("Model complexity:", bench.inspect_model_complexity())
        bench.benchmark_latency(imgsz=640, device="cpu", warmup_runs=5, test_runs=20)

    if args.step in ["all", "deploy"]:
        print("\n>>> [STEP 4/5] Deploying Model to Platform Runtime & Database...")
        from scripts.deploy_to_platform import deploy_model_to_platform
        best_p = BASE_DIR / "data" / "runs" / "exp_anpr_multisource_yolo26n" / "weights" / "best.pt"
        if not best_p.exists():
            best_p = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/yolo26n.pt")
        deploy_model_to_platform(best_p)

    if args.step in ["all", "paper"]:
        print("\n>>> [STEP 5/5] Generating Scientific Manuscript Draft (Markdown & Word docx)...")
        from scripts.generate_paper_doc import PaperDraftGenerator
        paper_gen = PaperDraftGenerator(output_dir=BASE_DIR / "paper")
        content = paper_gen.build_paper_content()
        paper_gen.generate_markdown(content)
        paper_gen.generate_docx(content)

    print("\n" + "=" * 65)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"Manuscript Draft (DOCX): {BASE_DIR / 'paper' / 'manuscript_draft.docx'}")
    print(f"Manuscript Draft (MD)  : {BASE_DIR / 'paper' / 'manuscript_draft.md'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
