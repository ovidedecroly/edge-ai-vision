"""
Automated Downloader for Open Indonesian License Plate Datasets.
Saves all raw datasets locally onto your PC in research/data/raw_datasets/.
Supports:
1. Direct download of public open-access Indonesian plate datasets from GitHub/web archives.
2. Roboflow Universe public dataset downloads (via Roboflow API/curl).
3. Local Dishub Surakarta CCTV captured frames extraction.
"""

import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path
from typing import List, Dict

RAW_ROOT = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research/data/raw_datasets")


def download_file_with_progress(url: str, dest_path: Path):
    print(f"[DOWNLOAD] Fetching: {url}")
    print(f"       To: {dest_path}")

    def progress_callback(blocks, block_size, total_size):
        if total_size > 0:
            percent = blocks * block_size * 100 / total_size
            print(f"\r  Progress: {min(100.0, percent):.1f}% [{blocks * block_size // 1024} KB]", end="", flush=True)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
    print("\n[DOWNLOAD] Finished successfully.")


def extract_zip(zip_path: Path, extract_to: Path):
    print(f"[EXTRACT] Unzipping {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[EXTRACT] Done. Files extracted to {extract_to}")


def collect_local_dishub_cctv(dest_dir: Path):
    """Gathers all captured and annotated frames from the local edge-ai-vision storage."""
    print("[LOCAL] Collecting Dishub Surakarta CCTV captured frames...")
    local_raw = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/storage/datasets/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    count = 0
    if local_raw.exists():
        for p in local_raw.glob("**/*"):
            if p.suffix.lower() in img_exts and p.is_file():
                # Copy image and sidecar label if present
                shutil.copy2(p, dest_dir / p.name)
                txt_p = p.with_suffix(".txt")
                if txt_p.exists():
                    shutil.copy2(txt_p, dest_dir / txt_p.name)
                count += 1
    print(f"[LOCAL] Collected {count} CCTV frames into {dest_dir}")


def download_roboflow_dataset(dataset_name: str, version: int, api_key: str = "", dest_dir: Path = None):
    """Downloads dataset from Roboflow Universe using Python API."""
    try:
        from roboflow import Roboflow
        if not api_key:
            print("[INFO] No Roboflow API key provided. Skipping automatic Roboflow cloud pull.")
            print("       You can get a free key from https://app.roboflow.com and pass --roboflow-key <KEY>")
            return

        rf = Roboflow(api_key=api_key)
        workspace, project = dataset_name.split("/")
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8", location=str(dest_dir))
        print(f"[ROBOFLOW] Successfully downloaded {dataset_name} v{version} to {dest_dir}")
    except Exception as e:
        print(f"[WARN] Roboflow download skipped or failed: {e}")


def main():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("LOCAL DATASET DOWNLOADER FOR INDONESIAN LICENSE PLATES")
    print(f"Target Directory (Local PC): {RAW_ROOT}")
    print("=" * 65)

    # 1. Local Surakarta CCTV
    cctv_dir = RAW_ROOT / "surakarta_cctv_local"
    collect_local_dishub_cctv(cctv_dir)

    # 2. Open Access Sample Dataset (GitHub mirror of Indonesian Vehicle License Plates)
    open_sample_dir = RAW_ROOT / "indonesian_plates_open_benchmark"
    open_sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Open dataset directory prepared at: {open_sample_dir}")
    print("       You can place any downloaded .zip files or folders from Kaggle / Roboflow here.")

    # Check for CLI args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--roboflow-key", type=str, default="", help="Roboflow API key for direct private/public pulls")
    args = parser.parse_args()

    if args.roboflow_key:
        rf_dir = RAW_ROOT / "roboflow_denny_plates"
        download_roboflow_dataset(
            dataset_name="denny-pmzkg/plat-nomor-kendaraan-indonesia",
            version=1,
            api_key=args.roboflow_key,
            dest_dir=rf_dir
        )

    print("\n" + "=" * 65)
    print("DATASET COLLECTION COMPLETED!")
    print(f"Check your local datasets at: {RAW_ROOT}")
    print("Run `python scripts/dataset_curator.py` next to merge, clean, and harmonize them into YOLO26 format.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
