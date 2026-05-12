"""
Unified Construction Site Safety Detection Pipeline
====================================================
Uses a SINGLE trained model (helmet_model_V2.pt) for:
  - Person detection (class 4)
  - Helmet detection (class 3)
  - Vest/Jacket detection (class 5)
  - Boots (class 0), Gloves (class 1), Goggles (class 2)

Plus MediaPipe PoseLandmarker for fall/pose detection,
and optical flow for falling object detection.

Usage:
    detector = SafetyDetector()
    annotated_frame, detections = detector.process_frame(frame)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode, PoseLandmark,
)
import time
import numpy as np
import logging
from ultralytics import YOLO
from pathlib import Path

logger = logging.getLogger("safety_detector")

# ─── Base directory for model paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ─── Detection Constants ──────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 3       # run AI every 3 frames for real-time speed
RESIZE_FOR_AI          = 640

# Person detection filters
PERSON_CONF_THRESHOLD  = 0.45    # higher confidence to avoid false positives
PPE_CONF_THRESHOLD     = 0.35    # PPE detection confidence
MIN_PERSON_AREA        = 1500    # minimum person bbox area in pixels (filters tiny misdetections)

# Fall detection constants
FALL_CONFIRM_SECS      = 1.5
MOTIONLESS_SECS        = 5.0
EDGE_MARGIN            = 0.12
MOVEMENT_THRESHOLD     = 18

# Falling object detection
FALLING_OBJ_SPEED      = 25

# PPE overlap matching (from user's helmet_detection_V2.py)
# Gear bbox must overlap person bbox by at least 40% of the gear's own area
GEAR_OVERLAP_THRESH    = 0.4

# Risk weights for scoring
RISK_WEIGHTS = {
    "no_helmet": 30,
    "no_vest": 20,
    "fall_confirmed": 50,
    "fall_possible": 25,
    "edge_danger": 40,
    "falling_object": 45,
    "motionless": 35,
}

# helmet_model_V2 class IDs
CLS_BOOTS   = 4
CLS_GLOVES  = 1
CLS_GOGGLES = 5
CLS_HELMET  = 0
CLS_PERSON  = 2
CLS_VEST    = 3


# ─── Per-Person Tracker (from Detection/fall_detection_V1_.py) ────────────────

class PersonTracker:
    """Tracks per-person state for fall and motionless detection."""

    def __init__(self):
        self.fall_start_time = None
        self.is_confirmed_fall = False
        self.last_center = None
        self.motionless_start = None

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update_movement(self, current_box):
        center = self.get_center(current_box)
        if self.last_center is None:
            self.last_center = center
            return False
        dx = abs(center[0] - self.last_center[0])
        dy = abs(center[1] - self.last_center[1])
        moved = (dx + dy) > MOVEMENT_THRESHOLD
        self.last_center = center
        if not moved:
            if self.motionless_start is None:
                self.motionless_start = time.time()
            elif time.time() - self.motionless_start >= MOTIONLESS_SECS:
                return True
        else:
            self.motionless_start = None
        return False

    def update_fall(self, is_falling_pose):
        if is_falling_pose:
            if self.fall_start_time is None:
                self.fall_start_time = time.time()
            elapsed = time.time() - self.fall_start_time
            if elapsed >= FALL_CONFIRM_SECS:
                self.is_confirmed_fall = True
                return "confirmed", elapsed
            return "possible", elapsed
        else:
            self.fall_start_time = None
            self.is_confirmed_fall = False
            return None, 0


# ─── PPE Overlap Logic (from Detection/helmet_detection_V2.py) ────────────────

def _overlap(person, gear, threshold=GEAR_OVERLAP_THRESH):
    """
    Check if gear bounding box overlaps with person bounding box.
    Returns True if gear box covers >= threshold of its own area inside person box.
    This is the proven logic from helmet_detection_V2.py.
    """
    x1 = max(person[0], gear[0])
    y1 = max(person[1], gear[1])
    x2 = min(person[2], gear[2])
    y2 = min(person[3], gear[3])

    if x2 < x1 or y2 < y1:
        return False
    intersection = (x2 - x1) * (y2 - y1)
    gear_area = (gear[2] - gear[0]) * (gear[3] - gear[1])
    if gear_area == 0:
        return False
    return (intersection / gear_area) > threshold


# ─── Fall Detection Helpers (from Detection/fall_detection_V1_.py) ─────────────

def _get_lm(landmarks, index, vis=0.4):
    lm = landmarks[index]
    return lm if lm.visibility >= vis else None


def _is_fall_position(landmarks):
    """
    Check if pose landmarks indicate a fall position.
    Logic from fall_detection_V1_.py, adapted for new MediaPipe Tasks API.

    Conditions for fall:
      1. All key landmarks (nose, shoulders, hips) must be visible
      2. Torso must be nearly flat (shoulder_y ≈ hip_y)
      3. Hips must be low in frame (hip_y > 0.6)
      4. Body must be roughly horizontal (nose_y ≈ feet_y)
    """
    nose = _get_lm(landmarks, PoseLandmark.NOSE)
    ls = _get_lm(landmarks, PoseLandmark.LEFT_SHOULDER)
    rs = _get_lm(landmarks, PoseLandmark.RIGHT_SHOULDER)
    lh = _get_lm(landmarks, PoseLandmark.LEFT_HIP)
    rh = _get_lm(landmarks, PoseLandmark.RIGHT_HIP)
    la = _get_lm(landmarks, PoseLandmark.LEFT_ANKLE)
    ra = _get_lm(landmarks, PoseLandmark.RIGHT_ANKLE)

    if not all([nose, ls, rs, lh, rh]):
        return False

    shoulder_y = (ls.y + rs.y) / 2
    hip_y = (lh.y + rh.y) / 2
    torso_flat = abs(shoulder_y - hip_y) < 0.1
    hip_low = hip_y > 0.6

    if la and ra:
        feet_y = (la.y + ra.y) / 2
        body_horiz = abs(nose.y - feet_y) < 0.35
    else:
        body_horiz = False

    return body_horiz and torso_flat and hip_low


# ─── Pose Skeleton Drawing ────────────────────────────────────────────────────

_POSE_CONNECTIONS = [
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
]


def _draw_pose_landmarks(frame, landmarks, h, w):
    """Draw pose skeleton on frame using OpenCV."""
    pts = {}
    for idx, lm in enumerate(landmarks):
        if lm.visibility >= 0.4:
            pts[idx] = (int(lm.x * w), int(lm.y * h))
    for a, b in _POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 165, 255), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 3, (0, 255, 255), -1)


# ─── Other Detection Helpers ──────────────────────────────────────────────────

def _is_near_edge(box, frame_h, frame_w):
    """Check if person is near the top edge (height danger)."""
    x1, y1, x2, y2 = box
    near_top = y1 < (frame_h * EDGE_MARGIN)
    if near_top:
        return True, "NEAR EDGE / HEIGHT DANGER"
    return False, ""


def _detect_falling_objects(current_gray, prev_gray, person_boxes):
    """Detect fast-moving objects falling above persons using optical flow."""
    if prev_gray is None:
        return []
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    alerts = []
    for pbox in person_boxes:
        px1, py1, px2, py2 = pbox
        pw = px2 - px1
        ph = py2 - py1
        above_y1 = max(0, py1 - ph)
        above_y2 = py1
        above_x1 = max(0, px1 - pw // 4)
        above_x2 = min(current_gray.shape[1], px2 + pw // 4)
        if above_y2 <= above_y1 or above_x2 <= above_x1:
            continue
        region_flow = flow[above_y1:above_y2, above_x1:above_x2]
        if region_flow.size == 0:
            continue
        vy_mean = np.mean(region_flow[..., 1])
        if vy_mean > FALLING_OBJ_SPEED:
            alerts.append((pbox, vy_mean))
    return alerts


def _get_tracker(box, trackers):
    """Find or create a PersonTracker for the given bounding box."""
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    for key in list(trackers.keys()):
        kx, ky = key
        if abs(cx - kx) < 80 and abs(cy - ky) < 80:
            trackers[(cx, cy)] = trackers.pop(key)
            return trackers[(cx, cy)]
    trackers[(cx, cy)] = PersonTracker()
    return trackers[(cx, cy)]


# ─── Main SafetyDetector Class ────────────────────────────────────────────────

class SafetyDetector:
    """
    Unified safety detection pipeline.

    Uses helmet_model_V2.pt (single model) for:
      - Person detection  (class 4, conf >= 0.45)
      - Helmet detection  (class 3, conf >= 0.35)
      - Vest detection    (class 5, conf >= 0.35)

    Plus MediaPipe PoseLandmarker for fall detection,
    and optical flow for falling object alerts.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(BASE_DIR / "models" / "helmet_model_V2.pt")

        logger.info(f"Loading safety model: {model_path}")
        self.model = YOLO(model_path)

        # MediaPipe Pose — new Tasks API
        pose_model_path = str(BASE_DIR / "models" / "pose_landmarker_lite.task")
        logger.info(f"Loading pose model: {pose_model_path}")
        self._init_pose_landmarker(pose_model_path)

        self.trackers = {}
        self.prev_gray = None
        self.frame_count = 0
        self.fps_history = []
        self.fps_display = 0.0

        self._last_person_boxes = []
        self._last_fall_pose = False
        self._last_falling_obj_alerts = []
        self._last_pose_landmarks = None
        self._last_helmets = []
        self._last_vests = []

        logger.info("SafetyDetector initialized successfully")

    def _init_pose_landmarker(self, pose_model_path=None):
        """Initialize or re-initialize the PoseLandmarker."""
        if pose_model_path is None:
            pose_model_path = str(BASE_DIR / "models" / "pose_landmarker_lite.task")
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=5,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(options)
        self._pose_timestamp_ms = 0

    def reset(self):
        """Reset all tracking state for a new video/session."""
        self.trackers = {}
        self.prev_gray = None
        self.frame_count = 0
        self.fps_history = []
        self.fps_display = 0.0
        self._last_person_boxes = []
        self._last_fall_pose = False
        self._last_falling_obj_alerts = []
        self._last_pose_landmarks = None
        self._last_helmets = []
        self._last_vests = []
        # Must re-create PoseLandmarker to reset internal timestamp state
        try:
            self._init_pose_landmarker()
        except Exception as e:
            logger.warning(f"Failed to re-create PoseLandmarker: {e}")

    def process_frame(self, frame):
        """
        Process a single video frame through all detection modules.

        Returns:
            annotated_frame: Frame with detection overlays drawn
            detections: List of dicts, one per detected person
        """
        self.frame_count += 1
        frame_start = time.time()
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # ── Single YOLO inference for ALL classes ──────────────────────
            results = self.model(frame, conf=PPE_CONF_THRESHOLD, verbose=False,
                                imgsz=RESIZE_FOR_AI)

            person_boxes = []
            helmets = []
            vests = []

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id == CLS_PERSON:
                    # Apply stricter confidence for person detection
                    if conf < PERSON_CONF_THRESHOLD:
                        continue
                    # Filter out tiny misdetections (must be a real person-sized box)
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < MIN_PERSON_AREA:
                        continue
                    person_boxes.append((x1, y1, x2, y2))
                elif cls_id == CLS_HELMET:
                    helmets.append((x1, y1, x2, y2))
                elif cls_id == CLS_VEST:
                    vests.append((x1, y1, x2, y2))

            self._last_person_boxes = person_boxes
            self._last_helmets = helmets
            self._last_vests = vests

            # ── MediaPipe Pose for fall detection ─────────────────────────
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self._pose_timestamp_ms += 33
                pose_result = self.pose_landmarker.detect_for_video(
                    mp_image, self._pose_timestamp_ms
                )
                fall_pose = False
                if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    landmarks = pose_result.pose_landmarks[0]
                    fall_pose = _is_fall_position(landmarks)
                    self._last_pose_landmarks = landmarks
                else:
                    self._last_pose_landmarks = None
                self._last_fall_pose = fall_pose
            except Exception as e:
                logger.debug(f"Pose detection skipped: {e}")
                self._last_pose_landmarks = None
                self._last_fall_pose = False

            # ── Optical Flow — Falling Objects ────────────────────────────
            falling_obj_alerts = _detect_falling_objects(
                gray, self.prev_gray, person_boxes
            )
            self._last_falling_obj_alerts = falling_obj_alerts
        else:
            # Reuse cached results on non-AI frames (fast passthrough)
            person_boxes = self._last_person_boxes
            fall_pose = self._last_fall_pose
            falling_obj_alerts = self._last_falling_obj_alerts
            helmets = self._last_helmets
            vests = self._last_vests

        # ── Draw skeleton if pose landmarks detected ──────────────────────
        if self._last_pose_landmarks:
            _draw_pose_landmarks(annotated_frame, self._last_pose_landmarks,
                                 height, width)

        falling_obj_persons = [a[0] for a in falling_obj_alerts]

        # ── Evaluate ONLY detected persons (no random objects) ────────────
        detections = []
        for idx, pbox in enumerate(person_boxes):
            x1, y1, x2, y2 = pbox
            tracker = _get_tracker(pbox, self.trackers)

            dangers = []
            highest_color = (0, 255, 0)  # green = safe

            # 1. PPE checks — using overlap logic from helmet_detection_V2.py
            has_helmet = any(_overlap(pbox, h) for h in helmets)
            has_vest = any(_overlap(pbox, v) for v in vests)
            if not has_helmet:
                dangers.append("NO HELMET")
                highest_color = (0, 0, 255)
            if not has_vest:
                dangers.append("NO VEST")
                highest_color = (0, 0, 255)

            # 2. Fall detection
            fall_status, fall_elapsed = tracker.update_fall(fall_pose)
            if fall_status == "confirmed":
                dangers.append(f"FALL DETECTED ({fall_elapsed:.1f}s)")
                highest_color = (0, 0, 255)
            elif fall_status == "possible":
                dangers.append(f"Possible Fall ({fall_elapsed:.1f}s)")
                if highest_color == (0, 255, 0):
                    highest_color = (0, 165, 255)

            # 3. Edge / height danger
            edge_danger, edge_msg = _is_near_edge(pbox, height, width)
            if edge_danger:
                dangers.append(edge_msg)
                highest_color = (0, 0, 255)

            # 4. Falling object
            is_falling_obj = pbox in falling_obj_persons
            if is_falling_obj:
                dangers.append("FALLING OBJECT!")
                highest_color = (0, 0, 255)

            # 5. Motionless / unconscious
            motionless = tracker.update_movement(pbox)
            if motionless:
                dangers.append("MOTIONLESS - CHECK PERSON")
                highest_color = (0, 0, 255)

            # Determine risk level
            if highest_color == (0, 0, 255):
                risk_level = "danger"
            elif highest_color == (0, 165, 255):
                risk_level = "warning"
            else:
                risk_level = "safe"

            # Draw person bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), highest_color, 2)

            # Draw stacked danger labels above person box
            label_list = dangers if dangers else ["SAFE"]
            for i, lbl in enumerate(label_list):
                lbl_size, baseline = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                offset = (len(label_list) - i) * (lbl_size[1] + 12)
                lbl_y = max(y1 - offset, lbl_size[1] + 10)
                bg_color = highest_color if lbl != "SAFE" else (0, 200, 0)
                cv2.rectangle(
                    annotated_frame,
                    (x1, lbl_y - lbl_size[1] - 4),
                    (x1 + lbl_size[0] + 6, lbl_y + baseline),
                    bg_color, cv2.FILLED,
                )
                cv2.putText(
                    annotated_frame, lbl, (x1 + 3, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

            detections.append({
                "person_id": idx,
                "bbox": [x1, y1, x2, y2],
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "fall_status": fall_status,
                "edge_danger": edge_danger,
                "falling_object": is_falling_obj,
                "motionless": motionless,
                "dangers": dangers,
                "risk_level": risk_level,
            })

        # Draw detected gear boxes (thin outlines)
        for hx1, hy1, hx2, hy2 in helmets:
            cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)
            cv2.putText(annotated_frame, "Helmet", (hx1, hy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        for vx1, vy1, vx2, vy2 in vests:
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 255, 255), 1)
            cv2.putText(annotated_frame, "Vest", (vx1, vy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Edge danger zone line
        danger_zone_y = int(height * EDGE_MARGIN)
        cv2.line(annotated_frame, (0, danger_zone_y), (width, danger_zone_y),
                 (0, 0, 255), 1)
        cv2.putText(annotated_frame, "-- EDGE DANGER ZONE --",
                    (10, danger_zone_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # FPS counter
        elapsed_ms = (time.time() - frame_start) * 1000
        if elapsed_ms > 0:
            self.fps_history.append(1000 / elapsed_ms)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            self.fps_display = sum(self.fps_history) / len(self.fps_history)

        cv2.putText(
            annotated_frame,
            f"FPS: {self.fps_display:.1f}  |  Workers: {len(person_boxes)}  |  AI Safety Monitor",
            (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
        )

        self.prev_gray = gray.copy()
        return annotated_frame, detections

    @staticmethod
    def get_risk_score(detections):
        """Calculate overall risk score (0-100) from detections."""
        if not detections:
            return 0.0
        total_weight = 0
        max_possible = len(detections) * sum(RISK_WEIGHTS.values())
        for det in detections:
            if not det["has_helmet"]:
                total_weight += RISK_WEIGHTS["no_helmet"]
            if not det["has_vest"]:
                total_weight += RISK_WEIGHTS["no_vest"]
            if det["fall_status"] == "confirmed":
                total_weight += RISK_WEIGHTS["fall_confirmed"]
            elif det["fall_status"] == "possible":
                total_weight += RISK_WEIGHTS["fall_possible"]
            if det["edge_danger"]:
                total_weight += RISK_WEIGHTS["edge_danger"]
            if det["falling_object"]:
                total_weight += RISK_WEIGHTS["falling_object"]
            if det["motionless"]:
                total_weight += RISK_WEIGHTS["motionless"]
        if max_possible == 0:
            return 0.0
        return min(100.0, (total_weight / max_possible) * 100)

    @staticmethod
    def get_stats(detections):
        """Calculate summary statistics from detections."""
        total = len(detections)
        if total == 0:
            return {"worker_count": 0, "helmet_compliance": 100.0,
                    "vest_compliance": 100.0, "danger_count": 0,
                    "warning_count": 0, "safe_count": 0}
        helmets = sum(1 for d in detections if d["has_helmet"])
        vests = sum(1 for d in detections if d["has_vest"])
        return {
            "worker_count": total,
            "helmet_compliance": round((helmets / total) * 100, 1),
            "vest_compliance": round((vests / total) * 100, 1),
            "danger_count": sum(1 for d in detections if d["risk_level"] == "danger"),
            "warning_count": sum(1 for d in detections if d["risk_level"] == "warning"),
            "safe_count": sum(1 for d in detections if d["risk_level"] == "safe"),
        }
