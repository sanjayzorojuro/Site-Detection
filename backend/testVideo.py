from ultralytics import YOLO
import cv2

#checking for GPU
import torch
print(torch.cuda.is_available(0))
print(torch.cuda.get_device_name())



model = YOLO("yolov8s.pt")

video_path = "X:\\Construction-Site\\backend\\testvid.mp4"  
cap = cv2.VideoCapture(video_path)

#getting the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# output video
out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print("Processing video... Press Q to stop")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame , conf = 0.15)

    # Draw detections
    annotated_frame = results[0].plot()

    # Write to output
    out.write(annotated_frame)

    # Show live
    cv2.imshow("Video Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved as output_video.mp4")