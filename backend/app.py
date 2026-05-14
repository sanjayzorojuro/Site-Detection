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
import smtplib
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from pydantic import BaseModel

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
    clear_violations, clear_all_data
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

# ─── Cache Busting ────────────────────────────────────────────────────────────
# Inject a version timestamp so CSS/JS URLs always bust browser cache
_CACHE_BUST = str(int(time.time()))
templates.env.globals["v"] = _CACHE_BUST

@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Prevent browser caching of CSS/JS static files during development."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") and (
        request.url.path.endswith(".css") or request.url.path.endswith(".js")
    ):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

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


# ─── Startup Event: Clear old data ────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    """Clear all stored violations and analytics on every server start."""
    logger.info("Server starting — clearing previous session data...")
    clear_all_data()
    logger.info("Previous session data cleared.")


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


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """Serve the about page."""
    return templates.TemplateResponse(request, "about.html")


@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    """Serve the contact page."""
    return templates.TemplateResponse(request, "contact.html")


# ─── Video Processing ────────────────────────────────────────────────────────

def _process_video(video_path: str, video_name: str):
    """Background thread: process video frames through the detection pipeline."""
    global state
    try:
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
            frame_start = time.time()

            # Resize large frames to keep pipeline fast
            h, w = frame.shape[:2]
            max_w = 960
            if w > max_w:
                scale = max_w / w
                frame = cv2.resize(frame, (max_w, int(h * scale)))

            annotated_frame, detections = det.process_frame(frame)

            # Encode frame as JPEG for streaming
            _, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
                                "timestamp": datetime.now().isoformat(),
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

            # Throttle to match real-time playback speed
            # Subtract actual processing time so the video plays at native FPS
            frame_time = time.time() - frame_start
            target_delay = 1.0 / video_fps
            sleep_time = target_delay - frame_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)

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

    except Exception as e:
        logger.error(f"Video processing CRASHED: {e}", exc_info=True)

    state.is_processing = False
    state.stop_flag = False
    state.current_frame = None
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
                            "timestamp": datetime.now().isoformat(),
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
    state.current_frame = None
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
    # Give the processing thread a moment to notice the flag, then force-clear
    state.current_frame = None
    state.is_processing = False
    return {"status": "stopped"}


@app.get("/api/video-feed")
async def video_feed():
    """MJPEG stream of the processed video with detection overlays."""
    async def generate():
        # Wait up to 10 seconds for first frame to appear
        waited = 0
        while state.is_processing and state.current_frame is None and waited < 10:
            await asyncio.sleep(0.1)
            waited += 0.1

        while state.is_processing:
            frame = state.current_frame
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
            await asyncio.sleep(0.025)  # ~40fps cap for smooth playback

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


# ─── Contact Form ─────────────────────────────────────────────────────────────

# ▸ Set your email credentials here or via environment variables:
#   CONTACT_EMAIL       = the Gmail address that SENDS the email
#   CONTACT_EMAIL_PASS  = Gmail App Password (NOT your login password)
#   CONTACT_RECIPIENT   = the email address that RECEIVES the messages
#
# To generate a Gmail App Password:
#   1. Enable 2-Step Verification on your Google account
#   2. Go to https://myaccount.google.com/apppasswords
#   3. Create an App Password and paste it below / in env var

CONTACT_EMAIL      = os.getenv("CONTACT_EMAIL", "sanjayraops17@gmail.com")
CONTACT_EMAIL_PASS = os.getenv("CONTACT_EMAIL_PASS", "96637761407483050218")
CONTACT_RECIPIENT  = os.getenv("CONTACT_RECIPIENT", CONTACT_EMAIL)


class ContactForm(BaseModel):
    name: str
    email: str
    subject: str
    message: str


@app.post("/api/contact")
async def contact_form(form: ContactForm):
    """Receive a contact form submission, store it, and email it."""
    # Store in MongoDB (or fallback)
    db = None
    try:
        from backend.database import get_db
        db = get_db()
    except Exception:
        pass

    doc = {
        "name": form.name,
        "email": form.email,
        "subject": form.subject,
        "message": form.message,
        "timestamp": datetime.now(),
        "read": False,
    }

    if db is not None:
        try:
            db.contact_messages.insert_one(doc)
        except Exception as e:
            logger.warning(f"Failed to store contact message: {e}")

    # Send email via SMTP (Gmail)
    email_sent = False
    if CONTACT_EMAIL_PASS != "your-app-password":
        try:
            msg = MIMEMultipart()
            msg["From"] = CONTACT_EMAIL
            msg["To"] = CONTACT_RECIPIENT
            msg["Subject"] = f"[SafeSite Contact] {form.subject}"

            body = (
                f"New message from SafeSite AI contact form:\n\n"
                f"Name:    {form.name}\n"
                f"Email:   {form.email}\n"
                f"Subject: {form.subject}\n\n"
                f"Message:\n{form.message}\n"
            )
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(CONTACT_EMAIL, CONTACT_EMAIL_PASS)
                server.send_message(msg)
            email_sent = True
            logger.info(f"Contact email sent to {CONTACT_RECIPIENT} from {form.email}")
        except Exception as e:
            logger.warning(f"Email send failed: {e}")

    return {
        "status": "received",
        "email_sent": email_sent,
        "message": "Thank you! Your message has been received."
            + (" We'll get back to you soon." if email_sent else
               " (Email delivery pending — message stored in database.)")
    }
