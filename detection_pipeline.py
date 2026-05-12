"""
Unified Construction Site Safety Detection Pipeline
====================================================
Combines: Person detection, PPE (helmet/vest), fall detection (MediaPipe),
edge/height danger, falling object detection (optical flow), motionless detection,
and heavy machinery proximity detection into a single importable module.

Usage:
    detector = SafetyDetector()
    annotated_frame, detections = detector.process_frame(frame)
"""

import cv2
import mediapipe as mp
import time
import numpy as np
import logging
from ultralytics import YOLO
from pathlib import Path

logger = logging.getLogger("safety_detector")

# ─── Base directory for model paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ─── Detection Constants ──────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 2
RESIZE_FOR_AI          = 640

FALL_CONFIRM_SECS      = 1.5
MOTIONLESS_SECS        = 5.0
EDGE_MARGIN            = 0.12
MOVEMENT_THRESHOLD     = 18
FALLING_OBJ_SPEED      = 25
GEAR_OVERLAP_THRESH    = 0.4

# COCO class IDs for heavy machinery detection
MACHINERY_CLASSES = {2: "car", 5: "bus", 7: "truck"}
PROXIMITY_THRESHOLD_RATIO = 0.15

# Risk weights for scoring
RISK_WEIGHTS = {
    "no_helmet": 30,
    "no_vest": 20,
    "fall_confirmed": 50,
    "fall_possible": 25,
    "edge_danger": 40,
    "falling_object": 45,
    "motionless": 35,
    "machinery_proximity": 30,
}


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


def _overlap(person, gear, threshold=GEAR_OVERLAP_THRESH):
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


def _get_lm(landmarks, index, vis=0.4):
    lm = landmarks[index]
    return lm if lm.visibility >= vis else None


def _is_fall_position(landmarks, mp_pose):
    nose = _get_lm(landmarks, mp_pose.PoseLandmark.NOSE)
    ls = _get_lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs = _get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    lh = _get_lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    rh = _get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    la = _get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    ra = _get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
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


def _is_near_edge(box, frame_h, frame_w):
    x1, y1, x2, y2 = box
    near_top = y1 < (frame_h * EDGE_MARGIN)
    if near_top:
        return True, "NEAR EDGE / HEIGHT DANGER"
    return False, ""


def _detect_falling_objects(current_gray, prev_gray, person_boxes):
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
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    for key in list(trackers.keys()):
        kx, ky = key
        if abs(cx - kx) < 80 and abs(cy - ky) < 80:
            trackers[(cx, cy)] = trackers.pop(key)
            return trackers[(cx, cy)]
    trackers[(cx, cy)] = PersonTracker()
    return trackers[(cx, cy)]


def _box_distance(box_a, box_b, frame_w):
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    return dist / frame_w


class SafetyDetector:
    """
    Unified safety detection pipeline.
    Combines YOLO person detection, YOLO PPE detection (helmet_model_V2),
    MediaPipe pose estimation, optical flow, edge danger, and machinery proximity.
    """

    def __init__(self, person_model_path=None, safety_model_path=None):
        if person_model_path is None:
            person_model_path = str(BASE_DIR / "models" / "yolov8n.pt")
        if safety_model_path is None:
            safety_model_path = str(BASE_DIR / "models" / "helmet_model_V2.pt")

        logger.info(f"Loading person model: {person_model_path}")
        self.person_model = YOLO(person_model_path)
        logger.info(f"Loading safety model: {safety_model_path}")
        self.safety_model = YOLO(safety_model_path)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

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
        self._last_machinery = []

        logger.info("SafetyDetector initialized successfully")

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
        self._last_machinery = []

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

        ai_height = int(height * RESIZE_FOR_AI / width)
        scale_x = width / RESIZE_FOR_AI
        scale_y = height / ai_height

        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            ai_frame = cv2.resize(frame, (RESIZE_FOR_AI, ai_height))

            # Person + Machinery Detection
            person_results = self.person_model(
                ai_frame, conf=0.4, verbose=False, imgsz=RESIZE_FOR_AI
            )
            person_boxes = []
            machinery_boxes = []
            for box in person_results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                if cls_id == 0:
                    person_boxes.append((x1, y1, x2, y2))
                elif cls_id in MACHINERY_CLASSES:
                    machinery_boxes.append((x1, y1, x2, y2, MACHINERY_CLASSES[cls_id]))
            self._last_person_boxes = person_boxes
            self._last_machinery = machinery_boxes

            # PPE Detection (helmet_model_V2)
            safety_results = self.safety_model(ai_frame, conf=0.3, verbose=False)
            helmets = []
            vests = []
            for box in safety_results[0].boxes:
                cls = int(box.cls[0])
                name = safety_results[0].names[cls].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                if "helmet" in name:
                    helmets.append((x1, y1, x2, y2))
                elif "jacket" in name or "vest" in name:
                    vests.append((x1, y1, x2, y2))
            self._last_helmets = helmets
            self._last_vests = vests

            # MediaPipe Pose
            rgb = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            fall_pose = False
            if result.pose_landmarks:
                fall_pose = _is_fall_position(result.pose_landmarks.landmark, self.mp_pose)
                self._last_pose_landmarks = result.pose_landmarks
            else:
                self._last_pose_landmarks = None
            self._last_fall_pose = fall_pose

            # Optical Flow — Falling Objects
            falling_obj_alerts = _detect_falling_objects(gray, self.prev_gray, person_boxes)
            self._last_falling_obj_alerts = falling_obj_alerts
        else:
            person_boxes = self._last_person_boxes
            fall_pose = self._last_fall_pose
            falling_obj_alerts = self._last_falling_obj_alerts
            helmets = self._last_helmets
            vests = self._last_vests
            machinery_boxes = self._last_machinery

        # Draw skeleton
        if self._last_pose_landmarks:
            self.mp_draw.draw_landmarks(
                annotated_frame, self._last_pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(0, 165, 255), thickness=2),
            )

        falling_obj_persons = [a[0] for a in falling_obj_alerts]

        # Build detections per person
        detections = []
        for idx, pbox in enumerate(person_boxes):
            x1, y1, x2, y2 = pbox
            tracker = _get_tracker(pbox, self.trackers)
            dangers = []
            highest_color = (0, 255, 0)

            has_helmet = any(_overlap(pbox, h) for h in helmets)
            has_vest = any(_overlap(pbox, v) for v in vests)
            if not has_helmet:
                dangers.append("NO HELMET")
                highest_color = (0, 0, 255)
            if not has_vest:
                dangers.append("NO VEST")
                highest_color = (0, 0, 255)

            fall_status, fall_elapsed = tracker.update_fall(fall_pose)
            if fall_status == "confirmed":
                dangers.append(f"FALL DETECTED ({fall_elapsed:.1f}s)")
                highest_color = (0, 0, 255)
            elif fall_status == "possible":
                dangers.append(f"Possible Fall ({fall_elapsed:.1f}s)")
                if highest_color == (0, 255, 0):
                    highest_color = (0, 165, 255)

            edge_danger, edge_msg = _is_near_edge(pbox, height, width)
            if edge_danger:
                dangers.append(edge_msg)
                highest_color = (0, 0, 255)

            is_falling_obj = pbox in falling_obj_persons
            if is_falling_obj:
                dangers.append("FALLING OBJECT!")
                highest_color = (0, 0, 255)

            motionless = tracker.update_movement(pbox)
            if motionless:
                dangers.append("MOTIONLESS - CHECK PERSON")
                highest_color = (0, 0, 255)

            near_machine = False
            for mbox in machinery_boxes:
                mx1, my1, mx2, my2, mname = mbox
                dist = _box_distance(pbox, (mx1, my1, mx2, my2), width)
                if dist < PROXIMITY_THRESHOLD_RATIO:
                    near_machine = True
                    dangers.append(f"TOO CLOSE TO {mname.upper()}")
                    highest_color = (0, 0, 255)
                    break

            if highest_color == (0, 0, 255):
                risk_level = "danger"
            elif highest_color == (0, 165, 255):
                risk_level = "warning"
            else:
                risk_level = "safe"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), highest_color, 2)
            label_list = dangers if dangers else ["SAFE"]
            for i, lbl in enumerate(label_list):
                lbl_size, baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                offset = (len(label_list) - i) * (lbl_size[1] + 12)
                lbl_y = max(y1 - offset, lbl_size[1] + 10)
                bg_color = highest_color if lbl != "SAFE" else (0, 200, 0)
                cv2.rectangle(annotated_frame, (x1, lbl_y - lbl_size[1] - 4),
                              (x1 + lbl_size[0] + 6, lbl_y + baseline), bg_color, cv2.FILLED)
                cv2.putText(annotated_frame, lbl, (x1 + 3, lbl_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({
                "person_id": idx,
                "bbox": [x1, y1, x2, y2],
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "fall_status": fall_status,
                "edge_danger": edge_danger,
                "falling_object": is_falling_obj,
                "motionless": motionless,
                "machinery_proximity": near_machine,
                "dangers": dangers,
                "risk_level": risk_level,
            })

        # Draw gear boxes
        for hx1, hy1, hx2, hy2 in helmets:
            cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)
            cv2.putText(annotated_frame, "Helmet", (hx1, hy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        for vx1, vy1, vx2, vy2 in vests:
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 255, 255), 1)
            cv2.putText(annotated_frame, "Vest", (vx1, vy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Draw machinery boxes
        for mx1, my1, mx2, my2, mname in machinery_boxes:
            cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (0, 140, 255), 2)
            cv2.putText(annotated_frame, mname.upper(), (mx1, my1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

        # Edge danger zone line
        danger_zone_y = int(height * EDGE_MARGIN)
        cv2.line(annotated_frame, (0, danger_zone_y), (width, danger_zone_y), (0, 0, 255), 1)
        cv2.putText(annotated_frame, "-- EDGE DANGER ZONE --", (10, danger_zone_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # FPS
        elapsed_ms = (time.time() - frame_start) * 1000
        if elapsed_ms > 0:
            self.fps_history.append(1000 / elapsed_ms)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            self.fps_display = sum(self.fps_history) / len(self.fps_history)

        cv2.putText(annotated_frame,
                    f"FPS: {self.fps_display:.1f}  |  Workers: {len(person_boxes)}  |  AI Safety Monitor",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

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
            if det.get("machinery_proximity"):
                total_weight += RISK_WEIGHTS["machinery_proximity"]
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
