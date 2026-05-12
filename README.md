# SafeSite AI — Construction Site Safety Monitoring System

AI-powered construction site safety monitoring with real-time hazard detection, violation recording, and analytics dashboard.

## Features

- **Helmet Detection** — YOLOv8-based PPE compliance checking
- **Vest Detection** — Reflective jacket verification via bounding box overlap
- **Fall Detection** — MediaPipe pose estimation for horizontal body detection
- **Edge/Height Danger** — Spatial zone analysis for workers near edges
- **Falling Object Detection** — Optical flow analysis for fast-moving objects above workers
- **Motionless Detection** — Flags unconscious or incapacitated workers
- **Machinery Proximity** — COCO-based heavy equipment proximity warnings

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI/ML | YOLOv8, MediaPipe, OpenCV |
| Backend | Python FastAPI, WebSocket |
| Frontend | HTML/CSS/JS, Three.js (3D homepage) |
| Database | MongoDB |
| Models | `helmet_model_V2.pt`, `yolov8n.pt` |

## Project Structure

```
Construction-Site/
├── run.py                    # Entry point — starts the server
├── detection_pipeline.py     # Unified SafetyDetector class
├── backend/
│   ├── app.py               # FastAPI server (routes, MJPEG stream, WebSocket)
│   └── database.py          # MongoDB database layer
├── website/
│   ├── templates/
│   │   ├── index.html        # 3D scrollable homepage
│   │   ├── livemonitor.html  # Live monitoring dashboard
│   │   ├── alert.html        # Violation records page
│   │   └── analytics.html    # Risk analytics page
│   └── assets/
│       ├── css/              # Page stylesheets
│       └── js/               # Page scripts + Three.js scene
├── models/                   # Pre-trained YOLO model weights
├── dataset/                  # Training datasets (YOLO format)
├── Detection/                # Original standalone detection scripts
└── requirements.txt
```

## Setup & Run

### Prerequisites

- Python 3.10+
- MongoDB running on `localhost:27017` (install from https://www.mongodb.com/try/download/community)
- GPU recommended (CUDA) for real-time performance

### Step 1: Install Dependencies

```bash
# Activate the existing virtual environment
.\venv\Scripts\activate        # Windows

# Install new dependencies
pip install fastapi "uvicorn[standard]" python-multipart websockets pymongo mediapipe
```

### Step 2: Start MongoDB

Make sure MongoDB is running:
```bash
# Windows — if installed as service, it auto-starts
# Or run manually:
mongod --dbpath C:\data\db
```

### Step 3: Run the Server

```bash
python run.py
```

The server starts at **http://localhost:8000**

### Step 4: Use the Application

1. Open **http://localhost:8000** — 3D animated homepage
2. Click **Launch Monitor** → Live monitoring page
3. Choose a video source:
   - **Upload Video** — select a construction site video
   - **Webcam** — use your system camera
   - **Demo Video** — uses a bundled test video
4. Watch real-time detection with alerts below the video
5. Visit **Alerts** page to review all recorded violations
6. Visit **Analytics** page for risk analysis and charts

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | 3D scrollable dark-mode landing page with Three.js |
| Live Monitor | `/monitor` | Video input, detection feed, real-time alerts |
| Alerts | `/alerts` | Violation history table with filtering & pagination |
| Analytics | `/analytics` | Risk charts, violation breakdown, processed videos |

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/upload-video` | POST | Upload video for processing |
| `/api/start-webcam` | POST | Start webcam feed |
| `/api/start-demo` | POST | Start demo with bundled video |
| `/api/stop` | POST | Stop current processing |
| `/api/video-feed` | GET | MJPEG detection stream |
| `/api/stats` | GET | Current session statistics |
| `/api/alerts` | GET | Session alerts |
| `/api/violations` | GET/DELETE | Database violations (CRUD) |
| `/api/analytics` | GET | Aggregated analytics |
| `/api/processed-videos` | GET | Processed video history |
| `/ws/detections` | WebSocket | Real-time detection events |

## Notes

- If MongoDB is not available, the system automatically falls back to in-memory storage (data won't persist across restarts)
- The system uses `helmet_model_V2.pt` for PPE detection
- Detection runs on every 2nd frame for performance; overlays are drawn on every frame
- Video is resized to 640px width for AI processing; display remains full resolution
