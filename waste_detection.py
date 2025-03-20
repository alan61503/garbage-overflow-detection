import cv2
import time
import smtplib
import torch
from ultralytics import YOLO
from roboflow import Roboflow
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# SMTP Email Credentials (Replace with actual credentials)
SMTP_SERVER = "sandbox.smtp.mailtrap.io"
SMTP_PORT = 2525
SMTP_USERNAME = "2fb7c1aa88789a"
SMTP_PASSWORD = "50e786b0dca5a3"  # Replace with actual password
SENDER_EMAIL = "alanchrisdisilva2@gmail.com"
RECEIVER_EMAIL = "abiyabiju050805@gmail.com"  # Change to actual receiver

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="rnjapVu9r2SSVpSyfPHs")  
project = rf.workspace().project("garbage-container-detection-sam7i-ktipk")  
dataset = project.version(2).download("yolov8")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Default YOLOv8 model

# Open webcam
cap = cv2.VideoCapture(0)  # Change to a video file path if needed

plt.ion()  # Enable interactive mode for Matplotlib
print("Press 'CTRL+C' to stop the detection...")

# Garbage detection parameters
garbage_count = 0
alert_threshold = 5  # Number of garbage detections before sending an alert
last_alert_time = 0  # Last time an alert was sent
alert_interval = 60  # Minimum time (in seconds) between alerts

def send_email_alert(image_path):
    """Sends an email alert when too much garbage is detected, including an image."""
    subject = "🚨 Garbage Overflow Detected!"
    body = "The system has detected excessive garbage. Immediate action is required!"

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(body, "plain"))

    # Attach the image
    with open(image_path, "rb") as img_file:
        img = MIMEImage(img_file.read(), name="garbage_detection.jpg")
        msg.attach(img)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure connection
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("✅ Email Alert Sent with Image!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Count garbage detections
        garbage_detected = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  # Ensure it's a float
                label = model.names[int(box.cls[0])]  # Get detected object name

                if conf > 0.5:  # Confidence threshold
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    garbage_detected += 1

        garbage_count += garbage_detected  # Update global count

        # Save the frame as an image for email alert
        image_path = "detected_garbage.jpg"
        cv2.imwrite(image_path, frame)

        # Send email alert if threshold is met
        if garbage_count >= alert_threshold and (time.time() - last_alert_time > alert_interval):
            send_email_alert(image_path)
            last_alert_time = time.time()
            garbage_count = 0  # Reset counter after alert

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
