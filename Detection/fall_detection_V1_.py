# module 1 
# need beter fps and better fall detection 

import cv2
import mediapipe as mp
import time
import numpy as np
from ultralytics import YOLO

# ─── Model Setup ───────────────────────────────────────────────────────────────
person_model = YOLO("models/yolov8n.pt")

mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ─── Video Setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture("X:\\Construction-Site\\maintestvid.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fps    = fps if fps > 0 else 25

out = cv2.VideoWriter(
    "output_danger_detection.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

cv2.namedWindow("Danger Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Danger Detection", 1280, 720)

# ─── Tuning Constants ──────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 2      # run AI every 2 frames, draw all frames
RESIZE_FOR_AI          = 640    # shrink frame for AI processing only

FALL_CONFIRM_SECS      = 1.5
MOTIONLESS_SECS        = 5.0
EDGE_MARGIN            = 0.12
MOVEMENT_THRESHOLD     = 18
FALLING_OBJ_SPEED      = 25
OBJ_OVERLAP_THRESH     = 0.25

# ─── Per-Person Tracker ────────────────────────────────────────────────────────
class PersonTracker:
    def __init__(self):
        self.fall_start_time     = None
        self.is_confirmed_fall   = False
        self.last_center         = None
        self.motionless_start    = None

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
                return 'confirmed', elapsed
            return 'possible', elapsed
        else:
            self.fall_start_time   = None
            self.is_confirmed_fall = False
            return None, 0

# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_lm(landmarks, index, vis=0.4):
    lm = landmarks[index]
    return lm if lm.visibility >= vis else None

def is_fall_position(landmarks):
    nose = get_lm(landmarks, mp_pose.PoseLandmark.NOSE)
    ls   = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs   = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    lh   = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    rh   = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    la   = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    ra   = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

    if not all([nose, ls, rs, lh, rh]):
        return False

    shoulder_y = (ls.y + rs.y) / 2
    hip_y      = (lh.y + rh.y) / 2
    torso_flat = abs(shoulder_y - hip_y) < 0.1
    hip_low    = hip_y > 0.6

    if la and ra:
        feet_y     = (la.y + ra.y) / 2
        body_horiz = abs(nose.y - feet_y) < 0.35
    else:
        body_horiz = False

    return body_horiz and torso_flat and hip_low

def is_near_edge(box, frame_h, frame_w):
    x1, y1, x2, y2 = box
    person_h = y2 - y1
    person_w = x2 - x1
    near_top  = y1 < (frame_h * EDGE_MARGIN)
    small_box = (person_h < frame_h * 0.25) and (person_w < frame_w * 0.15)
    if near_top:
        return True, "NEAR EDGE / HEIGHT DANGER"
    if small_box and near_top:
        return True, "PERSON AT HEIGHT"
    return False, ""

def detect_falling_objects(current_gray, prev_gray, person_boxes):
    if prev_gray is None:
        return []
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
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

def get_tracker(box, trackers):
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    for key in list(trackers.keys()):
        kx, ky = key
        if abs(cx - kx) < 80 and abs(cy - ky) < 80:
            trackers[(cx, cy)] = trackers.pop(key)
            return trackers[(cx, cy)]
    trackers[(cx, cy)] = PersonTracker()
    return trackers[(cx, cy)]

# ─── State Variables ───────────────────────────────────────────────────────────
trackers                = {}
prev_gray               = None
frame_count             = 0

# cached results reused on skipped frames
last_person_boxes       = []
last_fall_pose          = False
last_falling_obj_alerts = []
last_pose_landmarks     = None

# ─── AI frame scale factors (calculated once) ─────────────────────────────────
ai_height = int(height * RESIZE_FOR_AI / width)
scale_x   = width  / RESIZE_FOR_AI
scale_y   = height / ai_height

# ─── Main Loop ─────────────────────────────────────────────────────────────────
print("Processing... press Q to stop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count    += 1
    annotated_frame = frame.copy()
    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Run heavy AI only every N frames ─────────────────────────────────────
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        # Shrink frame for AI only — output stays full resolution
        ai_frame = cv2.resize(frame, (RESIZE_FOR_AI, ai_height))

        # ── YOLO Person Detection ─────────────────────────────────────────
        person_results = person_model(ai_frame, conf=0.4, verbose=False,
                                      imgsz=RESIZE_FOR_AI)
        person_boxes = []
        for box in person_results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # scale coords back to full resolution
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                person_boxes.append((x1, y1, x2, y2))
        last_person_boxes = person_boxes

        # ── MediaPipe Pose ────────────────────────────────────────────────
        rgb    = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        fall_pose = False
        if result.pose_landmarks:
            fall_pose              = is_fall_position(result.pose_landmarks.landmark)
            last_pose_landmarks    = result.pose_landmarks
        else:
            last_pose_landmarks    = None
        last_fall_pose = fall_pose

        # ── Optical Flow (Falling Objects) ────────────────────────────────
        falling_obj_alerts      = detect_falling_objects(gray, prev_gray, person_boxes)
        last_falling_obj_alerts = falling_obj_alerts

    else:
        # Skipped frame — reuse cached results, zero AI cost
        person_boxes       = last_person_boxes
        fall_pose          = last_fall_pose
        falling_obj_alerts = last_falling_obj_alerts

    # ── Always draw skeleton if landmarks available ───────────────────────────
    if last_pose_landmarks:
        mp_draw.draw_landmarks(
            annotated_frame,
            last_pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 165, 255), thickness=2)
        )

    falling_obj_persons = [a[0] for a in falling_obj_alerts]

    # ── Evaluate and Draw Each Person ─────────────────────────────────────────
    for pbox in person_boxes:
        x1, y1, x2, y2 = pbox
        tracker = get_tracker(pbox, trackers)

        dangers       = []
        highest_color = (0, 255, 0)  # green = safe

        # 1. Fall detection
        fall_status, fall_elapsed = tracker.update_fall(fall_pose)
        if fall_status == 'confirmed':
            dangers.append(f"FALL DETECTED ({fall_elapsed:.1f}s)")
            highest_color = (0, 0, 255)
        elif fall_status == 'possible':
            dangers.append(f"Possible Fall ({fall_elapsed:.1f}s)")
            if highest_color == (0, 255, 0):
                highest_color = (0, 165, 255)

        # 2. Edge / height danger
        edge_danger, edge_msg = is_near_edge(pbox, height, width)
        if edge_danger:
            dangers.append(edge_msg)
            highest_color = (0, 0, 255)

        # 3. Falling object
        if pbox in falling_obj_persons:
            dangers.append("FALLING OBJECT!")
            highest_color = (0, 0, 255)

        # 4. Motionless / unconscious
        motionless = tracker.update_movement(pbox)
        if motionless:
            dangers.append("MOTIONLESS - CHECK PERSON")
            highest_color = (0, 0, 255)

        # Draw person bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), highest_color, 2)

        # Draw stacked danger labels above person box
        label_list = dangers if dangers else ["SAFE"]
        for i, lbl in enumerate(label_list):
            lbl_size, baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            offset   = (len(label_list) - i) * (lbl_size[1] + 12)
            lbl_y    = max(y1 - offset, lbl_size[1] + 10)
            bg_color = highest_color if lbl != "SAFE" else (0, 200, 0)
            cv2.rectangle(
                annotated_frame,
                (x1, lbl_y - lbl_size[1] - 4),
                (x1 + lbl_size[0] + 6, lbl_y + baseline),
                bg_color, cv2.FILLED
            )
            cv2.putText(
                annotated_frame, lbl,
                (x1 + 3, lbl_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

    # ── Edge Danger Zone Line ─────────────────────────────────────────────────
    danger_zone_y = int(height * EDGE_MARGIN)
    cv2.line(annotated_frame, (0, danger_zone_y), (width, danger_zone_y), (0, 0, 255), 1)
    cv2.putText(annotated_frame, "-- EDGE DANGER ZONE --",
                (10, danger_zone_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ── FPS Counter on screen ─────────────────────────────────────────────────
    cv2.putText(annotated_frame,
                f"Processing every {PROCESS_EVERY_N_FRAMES} frames | AI size: {RESIZE_FOR_AI}px",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    prev_gray = gray.copy()

    out.write(annotated_frame)
    cv2.imshow("Danger Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done! Output saved to output_danger_detection.mp4")