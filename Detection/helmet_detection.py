from ultralytics import YOLO
import cv2

model = YOLO("models/helmet_model_V1.pt")

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("X:\\Construction-Site\\testvid.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("output_video.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps,
                      (width, height))

print("Processing... press Q to stop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    person_count = 0
    helmet_count = 0
    vest_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls == 4:  # person
            person_count += 1
        elif cls == 3:  # helmet
            helmet_count += 1
        elif cls == 5:  # vest
            vest_count += 1

    annotated_frame = results[0].plot()

    no_helmet = person_count > helmet_count
    no_vest = person_count > vest_count

    y = 45

    if person_count == 0:
        cv2.putText(annotated_frame, "No person detected", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
    else:

        if no_helmet:
            cv2.putText(annotated_frame,
                        f"WARNING: No Helmet! ({helmet_count}/{person_count})",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            y += 45

        if no_vest:
            cv2.putText(annotated_frame,
                        f"WARNING: No Vest! ({vest_count}/{person_count})",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,165,255), 3)
            y += 45

        if not no_helmet and not no_vest:
            cv2.putText(annotated_frame,
                        f"All Safe ({person_count} person(s))",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    out.write(annotated_frame)
    cv2.imshow("Safety Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done!")