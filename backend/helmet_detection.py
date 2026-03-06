from ultralytics import YOLO
import cv2

# Need both models now — person model finds people, helmet model finds gear
person_model = YOLO("../models/yolov8n.pt")
safety_model = YOLO("../models/helmet_model.pt")
# Classes: {0: 'Safety-Helmet', 1: 'Reflective-Jacket'}

cap = cv2.VideoCapture("X:\\Construction-Site\\testvid.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

print("Processing... press Q to stop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    person_results = person_model(frame, conf=0.4, verbose=False)
    safety_results = safety_model(frame, conf=0.4, verbose=False)

    # Count detected persons (class 0 in yolov8n is 'person')
    person_count = 0
    for box in person_results[0].boxes:
        if int(box.cls[0]) == 0:  # class 0 = person in coco dataset
            person_count += 1

    # Count detected safety gear
    helmet_count = 0
    jacket_count = 0
    for box in safety_results[0].boxes:
        class_name = safety_results[0].names[int(box.cls[0])]
        if class_name == "Safety-Helmet":
            helmet_count += 1
        if class_name == "Reflective-Jacket":
            jacket_count += 1

    # Draw safety gear boxes on frame (more useful to see)
    annotated_frame = safety_results[0].plot()

    # Logic: if more people than helmets/jackets → someone is missing gear
    no_helmet = person_count > helmet_count
    no_jacket = person_count > jacket_count

    # Show warnings
    y = 45
    if person_count == 0:
        cv2.putText(annotated_frame, "No person detected", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    else:
        if no_helmet:
            cv2.putText(annotated_frame, f"WARNING: No Helmet! ({helmet_count}/{person_count})", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            y += 45
        if no_jacket:
            cv2.putText(annotated_frame, f"WARNING: No Jacket! ({jacket_count}/{person_count})", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            y += 45
        if not no_helmet and not no_jacket:
            cv2.putText(annotated_frame, f"All Safe ({person_count} person(s))", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    out.write(annotated_frame)
    cv2.imshow("Safety Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done!")