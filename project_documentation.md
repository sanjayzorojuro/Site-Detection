 # SiteSpark — AI-Based Smart Construction Site Safety Monitoring System
## Complete Project Documentation

---

## 1. Project Overview

**SiteSpark** is an AI-powered construction site safety monitoring system that uses computer vision and deep learning to detect safety violations in real-time. The system processes video feeds (uploaded or live webcam) through a triple-model AI pipeline, identifies hazards such as missing helmets, missing vests, falls, motionless workers, falling objects, edge/height danger, and heavy machine proximity — then stores violations in MongoDB and displays them on a modern web dashboard.

### Key Capabilities
| # | Detection Module         | Technology Used                            |
|---|------------------------- |--------------------------------------------|
| 1 | Helmet Detection         | YOLOv8 Custom Model                        |
| 2 | Vest Detection           | YOLOv8 Custom Model                        |
| 3 | Fall Detection           | Google MediaPipe Pose                      |
| 4 | Motionless Detection     | Bounding Box Tracking                      |
| 5 | Falling Object Detection | OpenCV Optical Flow                        |
| 6 | Edge/Height Danger       | Spatial Position Analysis                  |
| 7 | Heavy Machine Proximity  | YOLOv8 Custom Model + Distance Calculation |

---

## 2. System Architecture

```
┌──────────────┐    ┌──────────────────────────────────────────────┐
│  User Browser │◄──►│           FastAPI Backend (app.py)            │
│  (HTML/CSS/JS)│    │  ┌────────────┐  ┌───────────────────────┐  │
│               │    │  │ Jinja2     │  │ detection_pipeline.py │  │
│  - index.html │    │  │ Templates  │  │  ┌─────────────────┐  │  │
│  - livemonitor│    │  └────────────┘  │  │ YOLOv8 Person   │  │  │
│  - alerts     │    │                  │  │ YOLOv8 PPE      │  │  │
│  - analytics  │    │  ┌────────────┐  │  │ YOLOv8 Machine  │  │  │
│  - about      │    │  │ WebSocket  │  │  │ MediaPipe Pose  │  │  │
│  - contact    │    │  │ /ws/detect │  │  │ Optical Flow    │  │  │
│               │    │  └────────────┘  │  └─────────────────┘  │  │
│  Three.js 3D  │    │                  └───────────────────────┘  │
│  Canvas Scene │    │  ┌────────────┐  ┌───────────────────────┐  │
│               │    │  │ MJPEG      │  │ database.py           │  │
│  WebSocket    │    │  │ Stream     │  │ MongoDB / Fallback    │  │
│  Client       │    │  └────────────┘  └───────────────────────┘  │
└──────────────┘    └──────────────────────────────────────────────┘
```

---



## 3. Technology Stack — Complete Library Documentation

### 3.1 Python Libraries

#### 3.1.1 Ultralytics (YOLOv8) — `ultralytics==8.4.19`

- **What it is:** State-of-the-art object detection framework by Ultralytics.
- **Why it's used:** YOLOv8 (You Only Look Once v8) provides real-time object detection with high accuracy. It can detect and classify objects in a single forward pass of the neural network, making it ideal for real-time video analysis.
- **How it's used in the project:**

  - **Model 1 — Person Detection (`yolov8n.pt`):** Pre-trained on the COCO dataset (80 classes). We use class 0 (person) to detect all workers in the frame. Confidence threshold: 0.45. Minimum bounding box area filter: 1500px to avoid                
false positives.

  - **Model 2 — PPE Detection (`helmet_model_V1.pt`):** Custom-trained YOLOv8 model trained on construction site images. Detects two classes: `Safety-Helmet` and `Reflective-Jacket`. Confidence threshold: 0.35.

  - **Model 3 — Machine Detection (`machine_model.pt`):** Custom-trained YOLOv8 model that detects 17 classes of heavy construction equipment (excavators, cranes, bulldozers, etc.). Confidence threshold: 0.40.

- **Key parameters:** `imgsz=480` (resized input for faster inference), `verbose=False`, inference runs every 4th frame (`PROCESS_EVERY_N_FRAMES=4`).




#### 3.1.2 OpenCV — `opencv-python==4.13.0.92`
- **What it is:** Open Source Computer Vision Library for image and video processing.
- **Why it's used:** Provides video capture, frame manipulation, image encoding, drawing functions, and optical flow computation.
- **How it's used:**
  - `cv2.VideoCapture()` — Opens video files and webcam streams
  - `cv2.resize()` — Resizes large frames to max 960px width for performance
  - `cv2.imencode('.jpg')` — Encodes frames as JPEG for MJPEG streaming
  - `cv2.cvtColor()` — Converts BGR↔RGB↔Gray color spaces
  - `cv2.rectangle()`, `cv2.putText()`, `cv2.circle()`, `cv2.line()` — Draws detection overlays (bounding boxes, labels, skeleton, danger zones)
  - `cv2.calcOpticalFlowFarneback()` — Computes dense optical flow between consecutive frames to detect falling objects


#### 3.1.3 MediaPipe — `mediapipe`
- **What it is:** Google's cross-platform ML framework for body pose estimation.
- **Why it's used:** Provides 33-point body landmark detection to analyze worker body posture for fall detection.
- **How it's used:**
  - `PoseLandmarker` initialized in `VIDEO` running mode with `num_poses=5`
  - Extracts landmarks: NOSE, LEFT/RIGHT_SHOULDER, LEFT/RIGHT_HIP, LEFT/RIGHT_ANKLE
  - Fall detection algorithm checks 4 conditions simultaneously:
    1. Torso nearly flat (shoulder_y ≈ hip_y, diff < 0.07)
    2. Hips low in frame (hip_y > 0.65)
    3. Body horizontal (nose_y ≈ feet_y, diff < 0.2)
    4. Body wider than tall (lying down aspect ratio)
  - Fall must persist for 3+ seconds before confirmed alert
  - Pose skeleton drawn on frame using `_POSE_CONNECTIONS` list


#### 3.1.4 PyTorch — `torch==2.5.1+cu121`
- **What it is:** Deep learning framework by Meta AI.
- **Why it's used:** Backend engine for YOLOv8 model inference. The `+cu121` build enables NVIDIA CUDA GPU acceleration.
- **How it's used:** Loaded automatically by Ultralytics. Handles tensor operations, model loading (`.pt` files), and GPU inference.


#### 3.1.5 TorchVision — `torchvision==0.20.1+cu121`
- **What it is:** Image processing utilities for PyTorch.
- **Why it's used:** Provides image transforms and model utilities required by the YOLO pipeline.


#### 3.1.6 NumPy — `numpy==2.3.5`
- **What it is:** Numerical computing library for Python.
- **Why it's used:** Handles array operations for image data, optical flow vectors, and bounding box calculations.
- **How it's used:** `np.mean()` computes average optical flow velocity in regions above workers to detect falling objects.

#### 3.1.7 FastAPI — `fastapi`
- **What it is:** Modern, high-performance Python web framework for building APIs.
- **Why it's used:** Provides async HTTP endpoints, WebSocket support, static file serving, and template rendering — all needed for the real-time monitoring dashboard.
- **How it's used:** See Section 4 (API Documentation) for full endpoint details.

#### 3.1.8 Uvicorn — `uvicorn[standard]`
- **What it is:** Lightning-fast ASGI server for Python.
- **Why it's used:** Serves the FastAPI application with support for WebSocket connections and async request handling.
- **How it's used:** Started via `run.py` with `host=localhost`, `port=8000`, `reload=True`.

#### 3.1.9 Jinja2 — `Jinja2==3.1.6`
- **What it is:** Python templating engine.
- **Why it's used:** Renders HTML templates with dynamic data (cache-busting version strings).
- **How it's used:** `templates.TemplateResponse()` serves each page. The `{{ v }}` variable is injected into CSS/JS URLs for cache busting.

#### 3.1.10 PyMongo — `pymongo`
- **What it is:** Official MongoDB driver for Python.
- **Why it's used:** Connects to MongoDB, performs CRUD operations on violation records.
- **How it's used:** See Section 5 (Database Documentation).

#### 3.1.11 Pydantic — (bundled with FastAPI)
- **What it is:** Data validation library using Python type hints.
- **Why it's used:** Validates incoming API request bodies (e.g., `ContactForm` model).
- **How it's used:** `class ContactForm(BaseModel)` ensures name, email, subject, message fields are present.

#### 3.1.12 python-multipart — `python-multipart`
- **What it is:** Streaming multipart form data parser.
- **Why it's used:** Required by FastAPI to handle file uploads (`UploadFile`).

#### 3.1.13 websockets — `websockets`
- **What it is:** WebSocket protocol implementation.
- **Why it's used:** Enables real-time bidirectional communication between server and browser for live detection stats and alert streaming.

#### 3.1.14 smtplib (Python Standard Library)
- **What it is:** Built-in SMTP email client.
- **Why it's used:** Sends contact form submissions via Gmail SMTP.
- **How it's used:** Connects to `smtp.gmail.com:587` with TLS, authenticates with App Password, sends formatted email.

#### 3.1.15 Other Supporting Libraries
| Library          | Version   | Purpose                                             |
|------------------|-----------|-----------------------------------------------------|
| `scipy`          | 1.17.1    | Scientific computing (used by MediaPipe internally) |
| `matplotlib`     | 3.10.8    | Plotting (development/training visualization)       |
| `pillow`         | 12.0.0    | Image processing (used by Ultralytics)              |
| `psutil`         | 7.2.2     | System monitoring                                   |
| `requests`       | 2.32.5    | HTTP requests (model downloads)                     |
| `PyYAML`         | 6.0.3     | YAML config parsing (Ultralytics config)            |
| `certifi`        | 2026.2.25 | SSL certificate verification                        |

### 3.2 Frontend Libraries

#### 3.2.1 Three.js — `r128` (CDN)
- **What it is:** 3D graphics library for the browser using WebGL.
- **Why it's used:** Creates an immersive 3D construction crane animation on the homepage hero section.
- **How it's used in `scene.js`:**
  - Builds a detailed construction crane model procedurally (base, tower lattice sections, operator cab, jib, counter-jib, trolley, hoist cable, hook)
  - Background cityscape with 6 buildings with lit windows
  - 300 floating particles (construction dust/sparks)
  - Scroll-responsive animation: crane rotates, trolley moves, hook drops, camera orbits
  - Responsive: adjusts scale/FOV for mobile vs desktop
  - Lighting: ambient, directional, rim, point lights with shadows
  - Materials: CAT yellow/black industrial theme with metalness/roughness

#### 3.2.2 Vanilla JavaScript (ES6+)
- **monitor.js** — Live monitor page controller (472 lines): handles video upload, webcam start, demo mode, WebSocket connection, MJPEG feed display, real-time stats updates, Canvas-based risk graph
- **alerts.js** — Alerts page controller (134 lines): fetches violations from DB API, renders paginated table, filtering by type/video, clear functionality
- **analytics.js** — Analytics page controller (160 lines): fetches aggregated analytics, renders bar charts and SVG donut charts, processed videos table
- **scene.js** — Three.js 3D scene (578 lines): full crane model with scroll animation

#### 3.2.3 Vanilla CSS
- 7 dedicated CSS files totaling ~94KB with: dark theme, glassmorphism effects, responsive layouts, hamburger menus, animations, gradients, custom properties

---

## 4. API Documentation

### 4.1 Page Routes (HTML)

| Method | Endpoint | Description | Template |
|---|---|---|---|
| GET | `/` | Homepage with 3D crane scene | `index.html` |
| GET | `/monitor` | Live monitoring dashboard | `livemonitor.html` |
| GET | `/alerts` | Violation history table | `alert.html` |
| GET | `/analytics` | Analytics dashboard with charts | `analytics.html` |
| GET | `/about` | About page with team info | `about.html` |
| GET | `/contact` | Contact form page | `contact.html` |

### 4.2 REST API Endpoints

#### Video Processing APIs

| Method | Endpoint | Description | Request | Response |
|---|---|---|---|---|
| POST | `/api/upload-video` | Upload and process a video | `multipart/form-data` with `file` | `{"status":"processing","video_name":"..."}` |
| POST | `/api/start-webcam` | Start webcam processing | None | `{"status":"processing","source":"webcam"}` |
| POST | `/api/start-demo` | Process bundled test video | `?video=testvid.mp4` | `{"status":"processing","video_name":"..."}` |
| POST | `/api/stop` | Stop current processing | None | `{"status":"stopped"}` |
| GET | `/api/video-feed` | MJPEG video stream | None | `multipart/x-mixed-replace` stream |

#### Data APIs

| Method | Endpoint | Description | Parameters | Response |
|---|---|---|---|---|
| GET | `/api/stats` | Current session stats | None | JSON with fps, risk_score, worker_count, etc. |
| GET | `/api/alerts` | Session alerts (in-memory) | `?limit=50` | `{"alerts":[...]}` |
| GET | `/api/violations` | DB violation records | `?video_name=&violation_type=&limit=200&skip=0` | `{"violations":[...],"total":N}` |
| DELETE | `/api/violations` | Clear violations | `?video_name=` (optional) | `{"status":"cleared"}` |
| GET | `/api/analytics` | Aggregated analytics | `?video_name=` (optional) | `{"total_violations":N,"by_type":{...},"by_risk":{...}}` |
| GET | `/api/processed-videos` | Processed video list | `?limit=50` | `{"videos":[...]}` |
| GET | `/api/test-videos` | List available test videos | None | `{"videos":[{"name":"...","size_mb":N}]}` |
| POST | `/api/contact` | Submit contact form | JSON body: name, email, subject, message | `{"status":"received","email_sent":bool}` |

### 4.3 WebSocket Endpoint

| Endpoint | Direction | Message Format |
|---|---|---|
| `ws://host/ws/detections` | Server→Client | `{"type":"stats","data":{...}}` every 1 second |
| | Server→Client | `{"type":"alert","data":{...}}` on new violation |

---

## 5. Database Documentation (MongoDB)

### 5.1 Connection
- **URI:** `mongodb://localhost:27017`
- **Database Name:** `construction_safety`
- **Connection Timeout:** 3000ms
- **Fallback:** In-memory Python lists when MongoDB is unavailable

### 5.2 Collections

#### `violations` Collection
| Field | Type | Description | Index |
|---|---|---|---|
| `_id` | ObjectId | Auto-generated unique ID | Primary |
| `video_name` | String | Source video filename | Yes |
| `violation_type` | String | e.g., "NO HELMET", "FALL DETECTED" | Yes |
| `risk_level` | String | "danger", "warning", or "safe" | — |
| `person_id` | Integer | Tracked person identifier | — |
| `bbox` | Array[4] | Bounding box [x1, y1, x2, y2] | — |
| `details` | String | Comma-separated danger descriptions | — |
| `frame_number` | Integer | Video frame number | — |
| `timestamp` | DateTime | When violation was recorded | Yes (DESC) |

#### `processed_videos` Collection
| Field | Type | Description | Index |
|---|---|---|---|
| `_id` | ObjectId | Auto-generated unique ID | Primary |
| `video_name` | String | Video filename | — |
| `date_processed` | DateTime | Processing timestamp | Yes (DESC) |
| `total_frames` | Integer | Number of frames processed | — |
| `total_violations` | Integer | Violations detected | — |
| `risk_level` | String | "danger", "warning", or "low" | — |
| `duration_seconds` | Float | Processing duration | — |

#### `contact_messages` Collection
| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | Auto-generated unique ID |
| `name` | String | Sender name |
| `email` | String | Sender email |
| `subject` | String | Message subject |
| `message` | String | Message body |
| `timestamp` | DateTime | Submission time |
| `read` | Boolean | Read status |

### 5.3 Aggregation Pipelines
The analytics module uses MongoDB aggregation pipelines:
1. **Violations by Type** — Groups violations by `violation_type`, counts occurrences, sorts descending
2. **Risk Distribution** — Groups violations by `risk_level`, counts danger/warning/safe

---

## 6. Detection Pipeline Documentation

### 6.1 Pipeline Flow (per frame)

```
Frame Input (BGR)
    │
    ▼ (every 4th frame)
┌───────────────────────────────────────────────────┐
│ Model 1: yolov8n.pt (COCO Person Detection)      │
│   → Filter class=0, conf≥0.45, area≥1500px       │
│   → Output: person bounding boxes                 │
├───────────────────────────────────────────────────┤
│ Model 2: helmet_model_V1.pt (PPE Detection)       │
│   → conf≥0.35                                     │
│   → Output: helmet boxes, vest boxes              │
├───────────────────────────────────────────────────┤
│ Model 3: machine_model.pt (Machine Detection)     │
│   → conf≥0.40                                     │
│   → Output: machine boxes + class names           │
├───────────────────────────────────────────────────┤
│ MediaPipe PoseLandmarker (Fall Detection)          │
│   → 33 body landmarks, VIDEO mode                 │
│   → Output: fall_pose boolean                     │
├───────────────────────────────────────────────────┤
│ Optical Flow (Falling Object Detection)            │
│   → Farneback method on grayscale frames          │
│   → Output: falling object alerts                 │
└───────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────┐
│ Per-Person Evaluation:                             │
│  1. PPE Check (overlap-based helmet/vest match)   │
│  2. Fall Detection (3s confirmation timer)        │
│  3. Edge/Height Danger (top 12% of frame)         │
│  4. Falling Object (optical flow above person)    │
│  5. Motionless Check (10s movement threshold)     │
│  6. Machine Proximity (120px distance threshold)  │
│  → Risk Level: danger / warning / safe            │
│  → Risk Score: weighted 0-100 scale               │
└───────────────────────────────────────────────────┘
    │
    ▼
Annotated Frame + Detection Results → MJPEG Stream + WebSocket + MongoDB
```

### 6.2 Risk Scoring Weights
| Violation | Weight |
|---|---|
| No Helmet | 30 |
| No Vest | 20 |
| Fall Confirmed | 50 |
| Fall Possible | 25 |
| Edge Danger | 40 |
| Falling Object | 45 |
| Motionless | 35 |
| Machine Proximity | 45 |

### 6.3 Key Algorithms

**PPE Overlap Matching:** A gear bounding box (helmet/vest) is assigned to a person if ≥40% of the gear's area overlaps with the person's bounding box.

**Fall Detection:** Uses 4 simultaneous conditions on MediaPipe pose landmarks — torso flatness, hip position, body horizontality, and aspect ratio. Must persist 3+ seconds to confirm.

**Motionless Detection:** Tracks bounding box center displacement across frames. If total movement < 18px for 10+ consecutive seconds, triggers alert.

**Machine Proximity:** Computes Euclidean gap between person and machine bounding boxes. If gap < 120px, triggers danger alert.

**Violation Deduplication:** 30-second cooldown per (person_id, violation_category) pair prevents duplicate database entries.

---

## 7. Frontend UI Documentation

### 7.1 Pages Overview

| Page | File | CSS | JS | Purpose |
|---|---|---|---|---|
| Homepage | `index.html` | `index.css` (30KB) | `scene.js` (22KB) | 3D crane hero, feature cards, modals, pipeline section |
| Live Monitor | `livemonitor.html` | `livemonitor.css` (14KB) | `monitor.js` (17KB) | Video feed, stats panel, risk graph, alerts feed |
| Alerts | `alert.html` | `alert.css` (8KB) | `alerts.js` (5KB) | Violation records table with filtering/pagination |
| Analytics | `analytics.html` | `analytics.css` (10KB) | `analytics.js` (6KB) | Bar charts, donut chart, processed videos table |
| About | `about.html` | `about.css` (13KB) | Inline | Team cards, feature modals, stats |
| Contact | `contact.html` | `contact.css` (8KB) | Inline | Contact form with SMTP email sending |

### 7.2 Design System
- **Theme:** Dark mode with CAT (Caterpillar) construction yellow/black
- **Primary Color:** `#f2c200` (CAT Yellow)
- **Background:** `#0a0a0a` (Near Black)
- **Font:** System fonts + Google Fonts (Rajdhani, Orbitron for UI elements)
- **Responsive:** Hamburger navigation on mobile, fluid grids, horizontal scroll tables

### 7.3 Real-time Communication Flow
```
Browser                          Server
   │                               │
   │── WebSocket Connect ─────────►│
   │                               │
   │◄── stats JSON (every 1s) ────│  (worker count, FPS, risk score)
   │◄── alert JSON (on event) ────│  (violation type, person ID)
   │                               │
   │── GET /api/video-feed ───────►│
   │◄── MJPEG stream ─────────────│  (continuous JPEG frames)
   │                               │
   │── GET /api/stats (fallback) ─►│  (HTTP polling every 2s)
   │◄── JSON response ────────────│
```

---

## 8. File Structure

```
Construction-Site/
├── run.py                      # Entry point — starts Uvicorn server
├── detection_pipeline.py       # Core AI detection engine (716 lines)
├── requirements.txt            # Python dependencies
├── PROJECT SYNOPSIS.txt        # Original project synopsis
│
├── backend/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application (715 lines)
│   └── database.py             # MongoDB data layer (259 lines)
│
├── website/
│   ├── templates/
│   │   ├── index.html          # Homepage (546 lines)
│   │   ├── livemonitor.html    # Live monitor (185 lines)
│   │   ├── alert.html          # Alerts history (116 lines)
│   │   ├── analytics.html      # Analytics dashboard (132 lines)
│   │   ├── about.html          # About page (275 lines)
│   │   └── contact.html        # Contact form (233 lines)
│   └── static/
│       ├── css/                # 7 CSS files (~94KB total)
│       ├── js/                 # 5 JS files (~52KB total)
│       └── images/team/        # Team member photos
│
├── models/
│   ├── yolov8n.pt              # COCO person detector (6.5MB)
│   ├── helmet_model_V1.pt      # Custom PPE detector (22.5MB)
│   ├── helmet_model_V2.pt      # PPE detector v2 (22.5MB)
│   ├── machine_model.pt        # Heavy machine detector (6.2MB)
│   └── pose_landmarker_lite.task  # MediaPipe pose model (5.7MB)
│
├── Detection/                  # Individual detection module scripts
│   ├── helmet_detection.py
│   ├── helmet_detection_V2.py
│   ├── fall_detection_V1_.py
│   ├── person_detection_V1.py
│   ├── combined_module_V1.py
│   └── machine_detection_V1.py
│
├── uploads/                    # Uploaded video storage
├── logs/                       # Application logs (safety.log)
└── *.mp4                       # Test videos for demo mode
```

---

## 9. How to Run the Project

### Prerequisites
- Python 3.10+
- MongoDB Server running on `localhost:27017`
- NVIDIA GPU with CUDA 12.1 (optional, for faster inference)

### Installation
```bash
# 1. Clone the repository
git clone <repository-url>
cd Construction-Site

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start MongoDB (separate terminal)
mongod

# 5. Run the application
python run.py
```

### Access
Open browser → `http://localhost:8000`

---

## 10. SMTP Email Configuration (Contact Form)

The contact form uses Gmail SMTP to forward messages:
- **Server:** `smtp.gmail.com:587` with STARTTLS
- **Authentication:** Gmail App Password (not regular password)
- **Setup:** Enable 2-Step Verification → Generate App Password at `myaccount.google.com/apppasswords`
- **Environment Variables:** `CONTACT_EMAIL`, `CONTACT_EMAIL_PASS`, `CONTACT_RECIPIENT`

---

## 11. Middleware & Performance Optimizations

| Optimization | Description |
|---|---|
| Cache Busting | Timestamp-based `?v=` parameter on CSS/JS URLs |
| No-Cache Headers | HTTP middleware prevents browser caching of static files during development |
| Frame Skipping | AI inference runs every 4th frame; cached results reused on intermediate frames |
| Frame Resizing | Input frames resized to 480px for AI, 960px max for display |
| Lazy Model Loading | Models loaded only on first detection request |
| Violation Cooldown | 30-second deduplication per (person, violation_type) pair |
| Session Alert Cap | In-memory alerts trimmed to 300 when exceeding 500 |
| JPEG Quality | Video: 80%, Webcam: 75% for bandwidth optimization |

---

## 12. Team Members

| Name | Role | Contributions |
|---|---|---|
| Sanjay | ML Engineer + Backend + 3D Designer | Detection pipeline, YOLO model training, 3D design, database, MediaPipe |
| Adithya | ML Engineer | YOLO/COCO model training, heavy machine detection, proximity system |
| Vignesh | UI/UX Designer + Frontend | User interface design, visual experience |
| Pratham | UI/UX Designer + Frontend + 3D | User interface design, visual experience |

---

*© 2026 SiteSpark — Smart Construction Safety System. All rights reserved.*
