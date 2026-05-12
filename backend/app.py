"""
FastAPI Backend for Construction Site Safety Monitoring
========================================================
Serves the web frontend, processes video/webcam feeds through the
unified detection pipeline, streams results via MJPEG + WebSocket,
and stores violations in MongoDB.
"""

import asyncio
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ─── Fix imports when running from project root ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from detection_pipeline import SafetyDetector
from backend.database import (
    insert_violation, get_violations, get_violation_count,
    get_analytics_summary, insert_processed_video, get_processed_videos,
    clear_violations
)

# ─── Logging Setup ────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "safety.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("backend")

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Construction Site Safety Monitor", version="1.0.0")

# ─── Static Files & Templates ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()
WEBSITE_DIR = BASE_DIR / "website"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(WEBSITE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBSITE_DIR / "templates"))

# ─── Global State ─────────────────────────────────────────────────────────────
detector = None  # Lazy-loaded SafetyDetector
processing_lock = threading.Lock()

# Current processing state
class ProcessingState:
    def __init__(self):
        self.is_processing = False
        self.current_frame = None  # Latest JPEG-encoded frame bytes
        self.latest_detections = []
        self.session_alerts = []  # Alerts for current session (in-memory for speed)
        self.session_stats = {}
        self.video_name = ""
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0.0
        self.source_type = ""  # "video" or "webcam"
        self.stop_flag = False
        self.risk_score = 0.0
        self.start_time = None

state = ProcessingState()

# WebSocket connections for real-time updates
ws_connections: list[WebSocket] = []


def get_detector():
    """Lazy-load the SafetyDetector (heavy model loading only once)."""
    global detector
    if detector is None:
        logger.info("Initializing SafetyDetector...")
        detector = SafetyDetector()
        logger.info("SafetyDetector ready.")
    return detector


# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Serve the 3D scrollable homepage."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request):
    """Serve the live monitoring page."""
    return templates.TemplateResponse(request, "livemonitor.html")


@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Serve the violation alerts history page."""
    return templates.TemplateResponse(request, "alert.html")


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Serve the analytics dashboard page."""
    return templates.TemplateResponse(request, "analytics.html")


# ─── Video Processing ────────────────────────────────────────────────────────

def _process_video(video_path: str, video_name: str):
    """Background thread: process video frames through the detection pipeline."""
    global state
    det = get_detector()
    det.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        state.is_processing = False
        return

    state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    state.video_name = video_name
    state.source_type = "video"
    state.start_time = time.time()
    state.frame_count = 0
    state.session_alerts = []

    # Track which violations have been reported to avoid duplicates
    last_violations_per_person = {}

    logger.info(f"Processing video: {video_name} ({state.total_frames} frames at {video_fps} fps)")

    while cap.isOpened() and not state.stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        state.frame_count += 1
        annotated_frame, detections = det.process_frame(frame)

        # Encode frame as JPEG for streaming
        _, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        state.current_frame = jpeg.tobytes()
        state.latest_detections = detections
        state.fps = det.fps_display
        state.risk_score = SafetyDetector.get_risk_score(detections)
        state.session_stats = SafetyDetector.get_stats(detections)

        # Generate alerts for new violations
        for det_item in detections:
            if det_item["dangers"]:
                pid = det_item["person_id"]
                current_dangers = tuple(sorted(det_item["dangers"]))
                prev_dangers = last_violations_per_person.get(pid)

                if current_dangers != prev_dangers:
                    last_violations_per_person[pid] = current_dangers
                    for danger in det_item["dangers"]:
                        alert = {
                            "id": str(uuid.uuid4()),
                            "type": danger,
                            "risk_level": det_item["risk_level"],
                            "person_id": pid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "frame_number": state.frame_count,
                            "video_name": video_name,
                        }
                        state.session_alerts.append(alert)
                        # Keep session alerts manageable
                        if len(state.session_alerts) > 500:
                            state.session_alerts = state.session_alerts[-300:]

                        # Store in MongoDB
                        try:
                            insert_violation(
                                video_name=video_name,
                                violation_type=danger,
                                risk_level=det_item["risk_level"],
                                person_id=pid,
                                bbox=det_item["bbox"],
                                details=", ".join(det_item["dangers"]),
                                frame_number=state.frame_count,
                            )
                        except Exception as e:
                            logger.warning(f"DB insert failed: {e}")

                        # Broadcast to WebSocket clients
                        _broadcast_alert(alert)
            else:
                last_violations_per_person.pop(det_item["person_id"], None)

        # Throttle to approximate real-time playback
        time.sleep(max(0.001, 1.0 / video_fps - 0.01))

    cap.release()

    # Record processed video in DB
    try:
        duration = time.time() - (state.start_time or time.time())
        insert_processed_video(
            video_name=video_name,
            total_frames=state.frame_count,
            total_violations=len(state.session_alerts),
            risk_level="danger" if state.risk_score > 50 else ("warning" if state.risk_score > 20 else "low"),
            duration_seconds=round(duration, 1),
        )
    except Exception as e:
        logger.warning(f"Failed to record processed video: {e}")

    state.is_processing = False
    state.stop_flag = False
    logger.info(f"Finished processing {video_name}: {state.frame_count} frames, {len(state.session_alerts)} alerts")


def _process_webcam():
    """Background thread: process webcam feed through the detection pipeline."""
    global state
    det = get_detector()
    det.reset()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        state.is_processing = False
        return

    state.video_name = "Webcam Live"
    state.source_type = "webcam"
    state.start_time = time.time()
    state.frame_count = 0
    state.total_frames = 0
    state.session_alerts = []

    last_violations_per_person = {}

    logger.info("Webcam feed started")

    while cap.isOpened() and not state.stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        state.frame_count += 1
        annotated_frame, detections = det.process_frame(frame)

        _, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        state.current_frame = jpeg.tobytes()
        state.latest_detections = detections
        state.fps = det.fps_display
        state.risk_score = SafetyDetector.get_risk_score(detections)
        state.session_stats = SafetyDetector.get_stats(detections)

        for det_item in detections:
            if det_item["dangers"]:
                pid = det_item["person_id"]
                current_dangers = tuple(sorted(det_item["dangers"]))
                prev_dangers = last_violations_per_person.get(pid)
                if current_dangers != prev_dangers:
                    last_violations_per_person[pid] = current_dangers
                    for danger in det_item["dangers"]:
                        alert = {
                            "id": str(uuid.uuid4()),
                            "type": danger,
                            "risk_level": det_item["risk_level"],
                            "person_id": pid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "frame_number": state.frame_count,
                            "video_name": "Webcam Live",
                        }
                        state.session_alerts.append(alert)
                        if len(state.session_alerts) > 500:
                            state.session_alerts = state.session_alerts[-300:]
                        try:
                            insert_violation(
                                video_name="Webcam Live",
                                violation_type=danger,
                                risk_level=det_item["risk_level"],
                                person_id=pid,
                                bbox=det_item["bbox"],
                                details=", ".join(det_item["dangers"]),
                                frame_number=state.frame_count,
                            )
                        except Exception as e:
                            logger.warning(f"DB insert failed: {e}")
                        _broadcast_alert(alert)
            else:
                last_violations_per_person.pop(det_item["person_id"], None)

    cap.release()
    state.is_processing = False
    state.stop_flag = False
    logger.info("Webcam feed stopped")


def _broadcast_alert(alert):
    """Send an alert to all connected WebSocket clients."""
    import json
    msg = json.dumps({"type": "alert", "data": alert})
    dead = []
    for ws in ws_connections:
        try:
            asyncio.run(ws.send_text(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections.remove(ws)


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and start processing it."""
    if state.is_processing:
        return JSONResponse({"error": "Already processing. Stop current session first."}, status_code=409)

    # Save uploaded file
    filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"Video uploaded: {filename} ({len(content)} bytes)")

    # Start processing in background thread
    state.is_processing = True
    state.stop_flag = False
    state.current_frame = None
    thread = threading.Thread(target=_process_video, args=(str(file_path), file.filename), daemon=True)
    thread.start()

    return {"status": "processing", "video_name": file.filename}


@app.post("/api/start-webcam")
async def start_webcam():
    """Start processing the webcam feed."""
    if state.is_processing:
        return JSONResponse({"error": "Already processing. Stop current session first."}, status_code=409)

    state.is_processing = True
    state.stop_flag = False
    state.current_frame = None
    thread = threading.Thread(target=_process_webcam, daemon=True)
    thread.start()

    return {"status": "processing", "source": "webcam"}


@app.post("/api/stop")
async def stop_processing():
    """Stop the current processing session."""
    state.stop_flag = True
    return {"status": "stopping"}


@app.get("/api/video-feed")
async def video_feed():
    """MJPEG stream of the processed video with detection overlays."""
    def generate():
        while state.is_processing or state.current_frame is not None:
            if state.current_frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    state.current_frame +
                    b"\r\n"
                )
            else:
                # Yield a tiny placeholder while waiting for first frame
                time.sleep(0.05)
            time.sleep(0.03)  # ~30fps cap for streaming

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stats")
async def get_stats():
    """Get current session statistics."""
    return {
        "is_processing": state.is_processing,
        "video_name": state.video_name,
        "source_type": state.source_type,
        "frame_count": state.frame_count,
        "total_frames": state.total_frames,
        "fps": round(state.fps, 1),
        "risk_score": round(state.risk_score, 1),
        "stats": state.session_stats,
        "alert_count": len(state.session_alerts),
        "elapsed": round(time.time() - state.start_time, 1) if state.start_time else 0,
    }


@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts from the current session."""
    alerts = state.session_alerts[-limit:]
    alerts.reverse()
    return {"alerts": alerts}


@app.get("/api/violations")
async def get_violations_api(video_name: str = None, violation_type: str = None,
                              limit: int = 200, skip: int = 0):
    """Get violation records from the database."""
    violations = get_violations(video_name, violation_type, limit, skip)
    # Serialize datetime objects
    for v in violations:
        if isinstance(v.get("timestamp"), datetime):
            v["timestamp"] = v["timestamp"].isoformat()
    total = get_violation_count(video_name)
    return {"violations": violations, "total": total}


@app.delete("/api/violations")
async def clear_violations_api(video_name: str = None):
    """Clear violation records from the database."""
    clear_violations(video_name)
    return {"status": "cleared"}


@app.get("/api/analytics")
async def get_analytics_api(video_name: str = None):
    """Get analytics summary data."""
    summary = get_analytics_summary(video_name)
    return summary


@app.get("/api/processed-videos")
async def get_processed_videos_api(limit: int = 50):
    """Get list of processed videos."""
    videos = get_processed_videos(limit)
    for v in videos:
        if isinstance(v.get("date_processed"), datetime):
            v["date_processed"] = v["date_processed"].isoformat()
    return {"videos": videos}


# ─── WebSocket for real-time detection events ─────────────────────────────────

@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """WebSocket endpoint for real-time detection updates."""
    await websocket.accept()
    ws_connections.append(websocket)
    logger.info(f"WebSocket connected. Total: {len(ws_connections)}")

    try:
        while True:
            # Send periodic stats updates
            import json
            stats_msg = json.dumps({
                "type": "stats",
                "data": {
                    "is_processing": state.is_processing,
                    "frame_count": state.frame_count,
                    "total_frames": state.total_frames,
                    "fps": round(state.fps, 1),
                    "risk_score": round(state.risk_score, 1),
                    "stats": state.session_stats,
                    "alert_count": len(state.session_alerts),
                    "elapsed": round(time.time() - state.start_time, 1) if state.start_time else 0,
                    "video_name": state.video_name,
                    "source_type": state.source_type,
                },
            })
            await websocket.send_text(stats_msg)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(ws_connections)}")
    except Exception as e:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
        logger.warning(f"WebSocket error: {e}")


# ─── Use existing test videos ─────────────────────────────────────────────────

@app.post("/api/start-demo")
async def start_demo(video: str = "testvid.mp4"):
    """Start processing one of the bundled test videos."""
    if state.is_processing:
        return JSONResponse({"error": "Already processing. Stop first."}, status_code=409)

    video_path = BASE_DIR / video
    if not video_path.exists():
        # Try other test videos
        for name in ["testvid5.mp4", "testvid2.mp4", "maintestvid.mp4"]:
            candidate = BASE_DIR / name
            if candidate.exists():
                video_path = candidate
                break
        else:
            return JSONResponse({"error": "No test video found."}, status_code=404)

    state.is_processing = True
    state.stop_flag = False
    state.current_frame = None
    thread = threading.Thread(
        target=_process_video, args=(str(video_path), video_path.name), daemon=True
    )
    thread.start()

    return {"status": "processing", "video_name": video_path.name}


@app.get("/api/test-videos")
async def list_test_videos():
    """List available test videos in the project directory."""
    videos = []
    for f in BASE_DIR.glob("*.mp4"):
        size_mb = f.stat().st_size / (1024 * 1024)
        videos.append({"name": f.name, "size_mb": round(size_mb, 1)})
    return {"videos": videos}
