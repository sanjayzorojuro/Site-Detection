from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") # downloading yolo


image_path = "X:\\Construction-Site\\backend\\test.jpg"  
img = cv2.imread(image_path)


results = model(img)  #Running the model


Final = results[0].plot()  # result image

# Show result
cv2.imshow("Detection Test", Final)
cv2.waitKey(0)    #waiting for a key to be pressed from keyboard
cv2.destroyAllWindows()   #close the window

# Save output
cv2.imwrite("output.jpg", Final)

print("Detection complete. Output saved as output.jpg")