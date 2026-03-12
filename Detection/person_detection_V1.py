from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")   # COCO model (person class = 0)

cap = cv2.VideoCapture("X:\\Construction-Site\\testvid.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "person_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print("Detecting persons... Press Q to stop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    annotated_frame = frame.copy()

    for box in results[0].boxes:
        cls = int(box.cls[0])

        # COCO class 0 = person
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                annotated_frame,
                f"Person {conf:.2f}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

    out.write(annotated_frame)
    cv2.imshow("Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")