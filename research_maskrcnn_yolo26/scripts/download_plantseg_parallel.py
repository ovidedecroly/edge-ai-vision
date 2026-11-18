"""
High-Speed Parallel Multi-Part Downloader for PlantSeg Dataset
Splits Zenodo download into 16 parallel chunks with HTTP Range requests.
"""

import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://zenodo.org/api/records/13958858/files/plantsegv2.zip/content"
DEST_FILE = Path("/media/ovide/DATA/PIT-UNS/edge-ai-vision/research_maskrcnn_yolo26/data/plantseg/plantsegv2.zip")
NUM_CHUNKS = 16


def get_file_size(url: str) -> int:
    resp = requests.head(url, headers={"User-Agent": "Wget/1.21.2"})
    resp.raise_for_status()
    return int(resp.headers.get("content-length", 0))


def download_chunk(url: str, start: int, end: int, part_path: Path):
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "Wget/1.21.2"
    }
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return part_path


def main():
    DEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = DEST_FILE.parent / "temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Querying total file size from Zenodo...")
    total_size = get_file_size(URL)
    print(f"[INFO] Total Size: {total_size / 1e6:.2f} MB ({total_size} bytes)")

    chunk_size = total_size // NUM_CHUNKS
    futures = []
    chunk_files = []

    t0 = time.time()
    print(f"[INFO] Launching {NUM_CHUNKS} parallel download workers...")

    with ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        for i in range(NUM_CHUNKS):
            start = i * chunk_size
            end = total_size - 1 if i == NUM_CHUNKS - 1 else (i + 1) * chunk_size - 1
            part_path = temp_dir / f"chunk_{i:02d}.part"
            chunk_files.append(part_path)
            
            # If already downloaded completely, skip
            if part_path.exists() and part_path.stat().st_size == (end - start + 1):
                print(f" Chunk {i:02d} already downloaded.")
                continue

            futures.append(executor.submit(download_chunk, URL, start, end, part_path))

        completed_count = 0
        for future in as_completed(futures):
            res_path = future.result()
            completed_count += 1
            elapsed = time.time() - t0
            print(f" [{completed_count}/{len(futures)}] Downloaded {res_path.name} in {elapsed:.1f}s")

    print(f"[INFO] All chunks downloaded in {time.time() - t0:.1f}s! Merging chunks into {DEST_FILE.name}...")
    with open(DEST_FILE, "wb") as outfile:
        for p in chunk_files:
            with open(p, "rb") as infile:
                while True:
                    buf = infile.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    outfile.write(buf)
            # Remove chunk
            p.unlink()

    temp_dir.rmdir()
    final_size = DEST_FILE.stat().st_size
    print(f"[SUCCESS] Download completed and verified: {DEST_FILE} ({final_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
