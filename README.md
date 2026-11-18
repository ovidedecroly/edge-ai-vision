# Edge AI Vision Research Platform for Intelligent CCTV Analytics
## Closed-Loop Research & Fine-Tuning Platform for YOLO26 on Dishub Surakarta CCTV

A production-grade, modular Edge-AI vision and research platform integrated with **96 CCTV ATCS Live Streams from Dinas Perhubungan Kota Surakarta** (`https://ccroom-dishub.surakarta.go.id/index.php/cctv`) alongside edge IP cameras (Hikvision DS-2CD1043G2-LIU).

Designed as an end-to-end **closed-loop research platform**:
```text
CCTV Stream ──► Frame Capture ──► YOLO26 Auto-Annotation ──► Human Review Studio ──► Dataset Versioning ──► YOLO26 Training (train.py) ──► Model Deployment ──► Live Evaluation & Monitoring
```

---

## 1. Key Research Features

1. **Closed-Loop Workflow**:
   - Continuous dataset acquisition directly from urban Indonesian traffic CCTV streams (Balaikota, Gladag, Purwosari, Panggung, Kerten, Gilingan, Balapan, UNS, etc.).
   - Automated pseudo-labeling with YOLO26.
   - Interactive human-in-the-loop review studio (draw, edit, resize, delete bbox, change class, approve).
   - Automated re-training and deployment.

2. **Dataset Acquisition & Quality Filter (Section 3, 4, 5)**:
   - Configurable sampling intervals (5s, 10s, 30s) and max storage quota enforcement.
   - 4-stage automated image quality assessment:
     * Blur detection via Laplacian variance (`var > 30.0`).
     * Brightness bounds check (`35 <= mean <= 230`).
     * Overexposure / glare filter (`glare_ratio < 0.20`).
     * Duplicate frame detection via 64-bit difference perceptual hashing (dHash Hamming distance `> 4`).
   - Standard hierarchical storage: `storage/datasets/raw/<camera_id>/YYYY-MM-DD/image001.jpg` with sidecar JSON metadata.

3. **YOLO26 Auto-Annotation & Review System (Section 6, 8, 9)**:
   - Primary model: YOLO26n pretrained.
   - Indonesian vehicle classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
   - Generates draft annotations in YOLO TXT format (`class xc yc w h conf`) and COCO JSON format.
   - Interactive Canvas studio with bounding box adjustment, class switching, and 1-click approval.

4. **Non-Random Temporal Dataset Splitting & Export (Section 10, 11, 12)**:
   - Enforces temporal and camera grouping (e.g. Day 1–5 train, Day 6 val, Day 7 test) to prevent video frame leakage.
   - 1-click export to standard Ultralytics YOLO26 directory structure:
     ```text
     storage/datasets/exported/<dataset_name>/
     ├── images/ (train, val, test)
     ├── labels/ (train, val, test)
     └── data.yaml
     ```

5. **YOLO26 Training Execution & Evaluation (Section 13, 14, 15)**:
   - Standalone CLI `train.py`:
     ```bash
     python train.py --model yolo26n.pt --dataset data.yaml --epochs 100 --imgsz 640 --batch 16
     ```
   - Background training process runner integrated into backend API (`/api/v1/experiments/train`).
   - Automated evaluation saving:
     * `runs/weights/best.pt` and `last.pt`
     * `runs/results.csv`
     * `runs/confusion_matrix.png`
     * `runs/PR_curve.png`
     * `runs/results.png` (training loss & mAP curves)
   - Real-time telemetry: Precision, Recall, mAP50, mAP50-95, F1-score, FPS, Latency (ms), GPU usage, and RAM.

6. **Two-Stage License Plate & OCR Preparation (Section 16, 17)**:
   - Architecture: Vehicle Detector ──► Vehicle Crop ──► Plate Detector ──► Plate Crop ──► OCR Preparation.
   - Plate Dataset Manager with CSV ground truth logging (`image, text, vehicle_class, confidence, date`).
   - Prepared export interfaces for PP-OCR, LPRNet, and CRNN (`ocr_train.txt`, `ocr_val.txt`).

7. **Live Monitoring & Research Dashboard (Section 18, 19)**:
   - Direct integration of all 96 CCTV ATCS Dishub Surakarta streams with district filtering.
   - Live bounding box overlay with class, confidence score, and tracking ID.
   - Real-time research panel (Current Model, FPS, Latency ms, Objects Detected, Vehicle Count breakdown).
   - Research Dashboard with 7 KPI cards and 4 statistical analytics charts.

8. **Model Registry & Deployment (Section 7, 20)**:
   - Support for `.pt`, `.onnx`, and TensorRT.
   - Automated CPU/CUDA smoke test validation before activation.
   - 1-click hot-swap model activation into live monitoring.
   - Rollback capability with full audit logging.

9. **Security & Flat Industrial UI (Section 21, 22)**:
   - No hardcoded passwords, environment variable based configuration.
   - Role-Based Access Control (Admin, Researcher, Annotator, Operator, Viewer).
   - Audit trail for all critical model, dataset, and camera operations.
   - Flat professional laboratory theme:
     * Background: `#F6F8FB`
     * Primary: `#173B65`
     * Secondary: `#147D83`
     * Success: `#238A57`
     * Warning: `#D58A16`
     * Danger: `#C94141`

---

## 2. Quickstart & Installation

### Prerequisites
- Linux (Ubuntu / Pop!_OS / Debian)
- Python 3.11 or 3.12 with virtual environment
- Node.js 20+ and npm
- NVIDIA GPU with CUDA (optional for GPU acceleration)

### Step 1: Backend Setup & Database Migration
```bash
cd edge-ai-vision/backend
# Activate virtual environment
source /home/ovide/venv/bin/activate

# Apply migrations
alembic upgrade head

# Run backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation is available at: `http://localhost:8000/docs`

### Step 2: Training a YOLO26 Model via CLI
```bash
python train.py --model yolo26n.pt --dataset storage/datasets/exported/dishub_traffic_yolo/data.yaml --epochs 100 --imgsz 640
```

### Step 3: Frontend Setup & Development
```bash
cd ../frontend
npm install
npm run dev
```
Open your browser at: `http://localhost:5173`
- Default Username: `admin`
- Default Password: `Admin@123456`

### Step 4: Run Tests
```bash
cd /media/ovide/DATA/PIT-UNS/edge-ai-vision
pytest tests
```
