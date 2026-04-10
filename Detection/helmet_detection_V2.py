# Main Running model 
 
from ultralytics import YOLO
import cv2


def overlap(person, gear, threshold=0.4):   # 0.4 means if the gear overlap person 40%  then its considerd as wearing the gear
    # this function is used to check if the helmet is In person or not by checking if the boxex overlap on each other.
    x1 = max(person[0], gear[0])
    y1 = max(person[1], gear[1])
    x2 = min(person[2], gear[2])
    y2 = min(person[3], gear[3])

    '''
    Person box:  x1=100, y1=50,  x2=300, y2=400
    Helmet box:  x1=120, y1=55,  x2=220, y2=130
    '''

    #negative width or hight will be false Ex:If two boxes are side by side with no touching, x2 will be less than x1, meaning they don't overlap.
    if x2 < x1 or y2 < y1:
        return False
    intersection = (x2 - x1) * (y2 - y1)
    gear_area = (gear[2] - gear[0]) * (gear[3] - gear[1])
    if gear_area == 0:
        return False
    return (intersection / gear_area) > threshold


# Load models
person_model = YOLO("models/yolov8n.pt")       # COCO person detector
safety_model = YOLO("models/helmet_model_V1.pt")  # helmet + vest detector

cap = cv2.VideoCapture("X:\\Construction-Site\\testvid5.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   #int convert it to integer since pixel cant be float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fps    = fps if fps > 0 else 25 # fallback if FPS undetectable

out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

cv2.namedWindow("Safety Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Safety Detection", 1280, 720)

print("Processing... press Q to stop")

while cap.isOpened():

    #cap.read returns two things. ret(return) = true if the frame was successfully read
    #frame = the actual image as a numpy array of pixel (height,width,3) 3 = bgr channels
    ret, frame = cap.read()
    if not ret:
        break

    # Run detections
    person_results = person_model(frame, conf=0.4)
    safety_results = safety_model(frame, conf=0.3)
    '''
        person_results → [person at (100,50,300,400) with 92% confidence]
        safety_results → [helmet at (120,55,220,130) with 78% confidence,
                          jacket at (110,150,290,380) with 65% confidence]
    '''

    annotated_frame = frame.copy()
    #store detected boxes
    persons = []
    helmets = []
    vests   = []

    # Collect person detections (COCO class 0 = person)
    for box in person_results[0].boxes:   #person_result[0] -> first image passed
        #gets class id and check if its person or not
        cls = int(box.cls[0])
        if cls == 0:                          
            x1, y1, x2, y2 = map(int, box.xyxy[0])  #box.xyxy[0] -> coordinates in format [left, top, right, bottom] as floats,map convert float to int
            persons.append((x1, y1, x2, y2))

    # Collect safety gear detections
    for box in safety_results[0].boxes:
        cls  = int(box.cls[0])
        name = safety_results[0].names[cls].lower()  #.names[cls] = converts class number to text. e.g. 0 → "Safety-Helmet"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if name == "safety-helmet":
            helmets.append((x1, y1, x2, y2))
        elif name == "reflective-jacket":
            vests.append((x1, y1, x2, y2))

    # Evaluate each detected person
    for (px1, py1, px2, py2) in persons:
        has_helmet = any(overlap((px1, py1, px2, py2), h) for h in helmets)
        has_vest   = any(overlap((px1, py1, px2, py2), v) for v in vests)

        # Build violation label
        violations = []
        if not has_helmet:
            violations.append("NO HELMET")
        if not has_vest:
            violations.append("NO Jacket")

        color = (0, 0, 255) if violations else (0, 255, 0)  # red or green
        label = " | ".join(violations) if violations else "SAFE"  #" | ".join(violations) combines list items: ["NO HELMET", "NO JACKET"] → "NO HELMET | NO JACKET"

        # Draw person bounding box
        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), color, 2)  # (px1, py1) = top-left corner  (px2, py2) = bottom-right corner , 2 = line thickness in pixels

        # Draw label with background for readability
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        #Calculates the Y position for the label, ensuring it doesn't go off the top of the screen.
        label_y = max(py1 - 10, label_size[1] + 10)
        cv2.rectangle(
            annotated_frame,
            (px1, label_y - label_size[1] - 5),
            (px1 + label_size[0], label_y + baseline),
            color,
            cv2.FILLED
        )
        cv2.putText(
            annotated_frame, label,
            (px1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    # Draw helmet boxes (blue)
    for (x1, y1, x2, y2) in helmets:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            annotated_frame, "Helmet",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    # Draw vest boxes (yellow)
    for (x1, y1, x2, y2) in vests:
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            annotated_frame, "Vest",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

    #saves the current annoted frame in output video file
    out.write(annotated_frame)
    cv2.imshow("Safety Detection", annotated_frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord("q") or key == 27:  # 27 = ESC
        break

cap.release()
out.release()  #flushes and finalizes the output video file. Without this, the output MP4 will be corrupted/unplayable
cv2.destroyAllWindows()
print("Done! Output saved to output_video_V2.mp4")