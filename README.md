⚡ SiteSpark

AI-Powered Construction Site Safety Monitoring System

Igniting safer construction sites with real-time AI hazard detection and intelligent analytics.

🚀 Overview

SiteSpark is an end-to-end computer vision system built to enhance safety in construction environments. It uses advanced AI models to monitor live video feeds, detect unsafe conditions, enforce PPE compliance, and provide actionable insights through a modern web dashboard.

The system supports live webcam monitoring, video uploads, and demo simulations, making it suitable for both real-world deployment and demonstrations.

✨ Key Features
🧠 AI-Based Safety Detection
Helmet Detection — PPE compliance using custom YOLOv8 model
Safety Vest Detection — Reflective jacket verification
Fall Detection — Pose estimation using MediaPipe
Motionless Detection — Detects unconscious or inactive workers
⚠️ Hazard Detection
Edge/Height Risk Detection — Identifies workers near dangerous zones
Falling Object Detection — Detects high-speed objects above workers
Machinery Proximity Alert — Warns when workers are too close to heavy equipment
📡 Real-Time Monitoring
Live video feed with bounding boxes and alerts
WebSocket-based real-time event streaming
MJPEG streaming for low-latency display
📊 Analytics & Insights
Violation history with filtering & pagination
Risk analytics and visual charts
Processed video tracking
Session-based statistics
🏗️ System Architecture
                ┌────────────────────┐
                │   Video Input      │
                │ Webcam / Upload    │
                └────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Detection Pipeline   │
              │ YOLOv8 + MediaPipe   │
              └────────┬─────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 Alerts Engine   Video Stream    Analytics Engine
        │              │              │
        ▼              ▼              ▼
   MongoDB        MJPEG Feed     Aggregation
        │
        ▼
   Dashboard UI (Frontend)
🧰 Tech Stack
Layer	Technology
AI/ML	YOLOv8, MediaPipe, OpenCV
Backend	FastAPI, WebSocket
Frontend	HTML, CSS, JavaScript, Three.js
Database	MongoDB
Streaming	MJPEG
Models	helmet_model_V2.pt, yolov8n.pt
📁 Project Structure
Construction-Site/
├── run.py
├── detection_pipeline.py
├── backend/
│   ├── app.py
│   └── database.py
├── website/
│   ├── templates/
│   │   ├── index.html
│   │   ├── livemonitor.html
│   │   ├── alert.html
│   │   └── analytics.html
│   └── assets/
│       ├── css/
│       └── js/
├── models/
├── dataset/
├── Detection/
└── requirements.txt
⚙️ Installation & Setup
🔹 Prerequisites
Python 3.10+
MongoDB running locally
(Optional) GPU with CUDA for real-time performance
🔹 Step 1: Clone Repository
git clone https://github.com/your-username/sitespark.git
cd sitespark
🔹 Step 2: Setup Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
🔹 Step 3: Install Dependencies
pip install -r requirements.txt
🔹 Step 4: Start MongoDB
mongod --dbpath C:\data\db

Falls back to in-memory storage if MongoDB is unavailable.

🔹 Step 5: Run the Server
python run.py
🌐 Usage

Open your browser:

👉 http://localhost:8000

Workflow:
Open homepage (3D animated UI)
Click Launch Monitor
Select input:
Upload Video
Webcam
Demo Video
View real-time detections & alerts
Explore:
Alerts Page → violation logs
Analytics Page → insights
🔌 API Endpoints
Endpoint	Method	Purpose
/api/upload-video	POST	Upload video
/api/start-webcam	POST	Start webcam
/api/start-demo	POST	Run demo
/api/stop	POST	Stop processing
/api/video-feed	GET	MJPEG stream
/api/stats	GET	Session stats
/api/alerts	GET	Alerts
/api/violations	GET/DELETE	DB operations
/api/analytics	GET	Analytics
/api/processed-videos	GET	History
/ws/detections	WebSocket	Live detections
⚡ Performance Notes
Processes every 2nd frame for optimization
Resizes video to 640px width for inference
Maintains full-resolution display
GPU recommended for real-time performance
🧪 Future Improvements
Multi-camera monitoring
Cloud deployment (AWS/GCP)
Worker identity tracking
Predictive risk scoring
Mobile app integration
🤝 Contributing
git checkout -b feature-name
git commit -m "Add feature"
git push origin feature-name
📜 License

MIT License

👨‍💻 Author

SiteSpark Team
Building smarter, safer construction sites ⚡
