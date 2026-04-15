import cv2
import mediapipe as mp
import time
import numpy as np
from ultralytics import YOLO

# ─── Model Setup ───────────────────────────────────────────────────────────────
person_model = YOLO("models/yolov8n.pt")  # COCO person detector

mp_pose   = mp.solutions.pose
pose      = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw   = mp.solutions.drawing_utils

# ─── Video Setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture("X:\\Construction-Site\\heightbuilding.mp4")

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

# ─── Constants ─────────────────────────────────────────────────────────────────
FALL_CONFIRM_SECS    = 1.5   # seconds in fall position before alerting
MOTIONLESS_SECS      = 5.0   # seconds without movement = possible unconscious
EDGE_MARGIN          = 0.12  # top 12% of frame = danger zone (edge of building)
MOVEMENT_THRESHOLD   = 18    # pixels — below this = considered not moving
FALLING_OBJ_SPEED    = 25    # pixels/frame — object moving this fast downward = danger
OBJ_OVERLAP_THRESH   = 0.25  # 25% of person box must overlap falling object

# ─── Per-Person Tracker ────────────────────────────────────────────────────────
# Tracks state for each detected person across frames using their box center
class PersonTracker:
    def __init__(self):
        self.fall_start_time      = None
        self.is_confirmed_fall    = False
        self.last_center          = None   # (x, y) center of person box last frame
        self.motionless_start     = None
        self.is_confirmed_motion  = False

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update_movement(self, current_box):
        """
        Checks if person has moved since last frame.
        Returns True if motionless too long.
        """
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
                return True   # motionless too long
        else:
            self.motionless_start = None  # reset if they move
        return False

    def update_fall(self, is_falling_pose):
        """
        Returns: 'confirmed', 'possible', or None
        """
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

# ─── Helper: get landmark if confident enough ──────────────────────────────────
def get_lm(landmarks, index, vis=0.4):
    lm = landmarks[index]
    return lm if lm.visibility >= vis else None

# ─── Helper: is fall pose ──────────────────────────────────────────────────────
def is_fall_position(landmarks):
    """
    3-check system:
    1. Head and feet at similar vertical height (body horizontal)
    2. Shoulders and hips at similar height (torso flat)
    3. Hips are low in frame (person is on ground)
    """
    nose   = get_lm(landmarks, mp_pose.PoseLandmark.NOSE)
    ls     = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs     = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    lh     = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
    rh     = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
    la     = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
    ra     = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

    if not all([nose, ls, rs, lh, rh]):
        return False

    shoulder_y = (ls.y + rs.y) / 2
    hip_y      = (lh.y + rh.y) / 2
    torso_flat = abs(shoulder_y - hip_y) < 0.1

    hip_low = hip_y > 0.6

    if la and ra:
        feet_y      = (la.y + ra.y) / 2
        body_horiz  = abs(nose.y - feet_y) < 0.35
    else:
        body_horiz  = False

    return body_horiz and torso_flat and hip_low

# ─── Helper: is person near edge ───────────────────────────────────────────────
def is_near_edge(box, frame_h, frame_w):
    """
    Checks if person box is in the danger zone:
    - Top edge: person is near top of frame (roof/ledge)
    - Also checks if person box is very small (far away = high up)
    Returns: (True/False, description)
    """
    x1, y1, x2, y2 = box
    person_h = y2 - y1
    person_w = x2 - x1

    # Top of person is in top EDGE_MARGIN of the frame
    near_top = y1 < (frame_h * EDGE_MARGIN)

    # Person box is small = person is far away = likely high up
    small_box = (person_h < frame_h * 0.25) and (person_w < frame_w * 0.15)

    if near_top:
        return True, "NEAR EDGE / HEIGHT DANGER"
    if small_box and near_top:
        return True, "PERSON AT HEIGHT"
    return False, ""

# ─── Helper: detect falling objects ────────────────────────────────────────────
prev_object_boxes = []  # stores non-person object boxes from previous frame

def detect_falling_objects(current_frame_gray, prev_frame_gray, person_boxes):
    """
    Uses optical flow to detect fast-moving objects coming from above
    toward a person's bounding box.
    Returns list of (person_box, speed) for persons being hit by something.
    """
    if prev_frame_gray is None:
        return []

    # Optical flow: tracks how every region of the image moved
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame_gray, current_frame_gray,
        None,
        0.5,   # pyramid scale
        3,     # pyramid levels
        15,    # window size
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0      # flags
    )

    alerts = []
    for pbox in person_boxes:
        px1, py1, px2, py2 = pbox
        pw = px2 - px1
        ph = py2 - py1

        # Check the region ABOVE the person (where objects would fall from)
        above_y1 = max(0, py1 - ph)          # one person-height above
        above_y2 = py1                        # top of person
        above_x1 = max(0, px1 - pw // 4)     # slightly wider than person
        above_x2 = min(current_frame_gray.shape[1], px2 + pw // 4)

        if above_y2 <= above_y1 or above_x2 <= above_x1:
            continue

        # Get the flow (movement) in that region above the person
        region_flow = flow[above_y1:above_y2, above_x1:above_x2]
        if region_flow.size == 0:
            continue

        # vy = vertical movement speed (positive = moving downward)
        vy_mean = np.mean(region_flow[..., 1])

        # Something is moving downward fast toward this person
        if vy_mean > FALLING_OBJ_SPEED:
            alerts.append((pbox, vy_mean))

    return alerts

# ─── Person Trackers Store ─────────────────────────────────────────────────────
# Key = approximate center tuple, Value = PersonTracker object
trackers = {}

def get_tracker(box):
    """
    Match a detected person box to an existing tracker by proximity,
    or create a new one if no match found.
    """
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2

    for key in trackers:
        kx, ky = key
        if abs(cx - kx) < 80 and abs(cy - ky) < 80:  # within 80px = same person
            # Update key position
            trackers[(cx, cy)] = trackers.pop(key)
            return trackers[(cx, cy)]

    # New person
    trackers[(cx, cy)] = PersonTracker()
    return trackers[(cx, cy)]

# ─── Main Loop ─────────────────────────────────────────────────────────────────
prev_gray = None
print("Processing... press Q to stop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Person Detection (YOLO) ───────────────────────────────────────────────
    person_results = person_model(frame, conf=0.4, verbose=False)
    person_boxes   = []

    for box in person_results[0].boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

    # ── Pose Estimation (MediaPipe) ───────────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    fall_pose = False
    if result.pose_landmarks:
        mp_draw.draw_landmarks(
            annotated_frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0,165,255), thickness=2)
        )
        fall_pose = is_fall_position(result.pose_landmarks.landmark)

    # ── Falling Object Detection ──────────────────────────────────────────────
    falling_obj_alerts = detect_falling_objects(gray, prev_gray, person_boxes)
    falling_obj_persons = [a[0] for a in falling_obj_alerts]

    # ── Evaluate Each Person ──────────────────────────────────────────────────
    for pbox in person_boxes:
        x1, y1, x2, y2 = pbox
        tracker = get_tracker(pbox)

        dangers        = []   # list of danger strings for this person
        highest_color  = (0, 255, 0)  # green default

        # ── 1. Fall / Slip Detection ──────────────────────────────────────
        fall_status, fall_elapsed = tracker.update_fall(fall_pose)
        if fall_status == 'confirmed':
            dangers.append(f"FALL DETECTED ({fall_elapsed:.1f}s)")
            highest_color = (0, 0, 255)   # red
        elif fall_status == 'possible':
            dangers.append(f"Possible Fall ({fall_elapsed:.1f}s)")
            if highest_color == (0, 255, 0):
                highest_color = (0, 165, 255)  # orange

        # ── 2. Edge / Height Danger ───────────────────────────────────────
        edge_danger, edge_msg = is_near_edge(pbox, height, width)
        if edge_danger:
            dangers.append(edge_msg)
            highest_color = (0, 0, 255)   # red

        # ── 3. Falling Object ─────────────────────────────────────────────
        if pbox in falling_obj_persons:
            dangers.append("FALLING OBJECT!")
            highest_color = (0, 0, 255)   # red

        # ── 4. Motionless / Unconscious ───────────────────────────────────
        motionless = tracker.update_movement(pbox)
        if motionless:
            dangers.append("MOTIONLESS - CHECK PERSON")
            highest_color = (0, 0, 255)   # red

        # ── Draw Person Box ───────────────────────────────────────────────
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), highest_color, 2)

        # ── Draw All Danger Labels stacked above person box ───────────────
        label_list = dangers if dangers else ["SAFE"]
        for i, lbl in enumerate(label_list):
            lbl_size, baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            offset  = (len(label_list) - i) * (lbl_size[1] + 12)
            lbl_y   = max(y1 - offset, lbl_size[1] + 10)
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

    # ── Edge Danger Zone Line (visual guide) ──────────────────────────────────
    danger_zone_y = int(height * EDGE_MARGIN)
    cv2.line(annotated_frame, (0, danger_zone_y), (width, danger_zone_y), (0, 0, 255), 1)
    cv2.putText(annotated_frame, "-- EDGE DANGER ZONE --",
                (10, danger_zone_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    prev_gray = gray.copy()

    out.write(annotated_frame)
    cv2.imshow("Danger Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done! Output saved to output_danger_detection.mp4")