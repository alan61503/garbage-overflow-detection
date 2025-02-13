import cv2
import torch
from ultralytics import YOLO
from roboflow import Roboflow
import matplotlib.pyplot as plt

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="rnjapVu9r2SSVpSyfPHs")  
project = rf.workspace().project("garbage-detection-h4vqo-r4vd1")  
dataset = project.version(1).download("yolov8")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Change to a video file path if needed

plt.ion()  # Enable interactive mode for Matplotlib
print("Press Enter to stop the detection...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference
        results = model(frame)

        # Draw detections on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  # Ensure it's a float
                label = model.names[int(box.cls[0])]

                if conf > 0.5:  # Confidence threshold
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame using Matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv8 Waste Detection")
        plt.axis("off")
        plt.pause(0.001)  # Update frame non-blockingly
        plt.clf()  # Clear previous frame to avoid overlay

except KeyboardInterrupt:
    print("\nDetection stopped manually.")

finally:
    cap.release()
    plt.close()  # Close Matplotlib figure
    print("Program terminated.")
