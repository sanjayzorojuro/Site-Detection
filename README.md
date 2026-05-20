# ⚡ SiteSpark  
**AI-Powered Construction Site Safety Monitoring System**

> Igniting safer construction sites with real-time AI hazard detection and intelligent analytics.

---

## 🚀 Overview

**SiteSpark** is an end-to-end computer vision system designed to improve safety in construction environments. It leverages AI to monitor live video feeds, detect hazards, enforce PPE compliance, and provide actionable insights through an interactive dashboard.

The system supports:
- 📷 Live webcam monitoring  
- 📁 Video uploads  
- 🎬 Demo simulations  
---
**Deployed At **

>  https://sitespark-swart.vercel.app/

---

## ✨ Key Features

### 🧠 AI-Based Detection
- Helmet Detection (YOLOv8 PPE model)
- Safety Vest Detection
- Fall Detection (MediaPipe pose estimation)
- Motionless Worker Detection

### ⚠️ Hazard Detection
- Edge/Height Risk Detection
- Falling Object Detection (optical flow)
- Machinery Proximity Alerts

### 📡 Real-Time Monitoring
- Live video stream with bounding boxes
- WebSocket-based real-time alerts
- MJPEG low-latency streaming

### 📊 Analytics & Insights
- Violation history with filters
- Risk analysis charts
- Processed video tracking
- Session statistics

---

## 🏗️ System Architecture

```
Video Input (Webcam / Upload)
            │
            ▼
Detection Pipeline (YOLOv8 + MediaPipe)
            │
   ┌────────┼────────┐
   ▼        ▼        ▼
Alerts   Streaming  Analytics
   │        │        │
   ▼        ▼        ▼
MongoDB   MJPEG    Dashboard UI
```

---

## 🧰 Tech Stack

| Layer      | Technology |
|-----------|-----------|
| AI/ML     | YOLOv8, MediaPipe, OpenCV |
| Backend   | FastAPI, WebSocket |
| Frontend  | HTML, CSS, JavaScript, Three.js |
| Database  | MongoDB |
| Streaming | MJPEG |
| Models    | helmet_model_V2.pt, yolov8n.pt |

---

## 📁 Project Structure

```
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
```

---

## ⚙️ Installation & Setup

### 🔹 Prerequisites
- Python 3.10+
- MongoDB running locally
- (Optional) GPU with CUDA

---

### 🔹 Clone Repository

```bash
git clone https://github.com/your-username/sitespark.git
cd Site-Detection
```

---

### 🔹 Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

---

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 Start MongoDB

```bash
mongod --dbpath C:\data\db
```

> If MongoDB is not running, the system falls back to in-memory storage.

---

### 🔹 Run the Application

```bash
python run.py
```

---

## 🌐 Usage

Open your browser:

http://localhost:8000

### Workflow:
1. Open homepage  
2. Click **Launch Monitor**  
3. Select input:
   - Upload Video  
   - Webcam  
   - Demo Video  
4. View real-time detections & alerts  
5. Visit:
   - `/alerts` → violation logs  
   - `/analytics` → insights  

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|------------|
| /api/upload-video | POST | Upload video |
| /api/start-webcam | POST | Start webcam |
| /api/start-demo | POST | Run demo |
| /api/stop | POST | Stop processing |
| /api/video-feed | GET | MJPEG stream |
| /api/stats | GET | Session stats |
| /api/alerts | GET | Alerts |
| /api/violations | GET/DELETE | Database operations |
| /api/analytics | GET | Analytics |
| /api/processed-videos | GET | Video history |
| /ws/detections | WebSocket | Real-time detections |

---

## ⚡ Performance Notes

- Processes every 2nd frame for optimization  
- Resizes video to 640px width for inference  
- Maintains full-resolution display  
- GPU recommended for real-time performance  

---

## 🧪 Future Enhancements

- Multi-camera monitoring  
- Cloud deployment (AWS/GCP)  
- Worker identity tracking  
- Predictive risk scoring  
- Mobile app integration  

---

## 🤝 Contributing

```bash
git checkout -b feature-name
git commit -m "Add feature"
git push origin feature-name
```

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Sanjay**  
Building smarter, safer construction sites ⚡
