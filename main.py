import cv2
import cvzone
import math
import requests
from ultralytics import YOLO

# LINE Notify Token
LINE_NOTIFY_TOKEN = "BSRDArBIGehIPYi4tqLNvwpkHVcO8t3IvIcMkdo9KHT"

def send_line_notify(message, image_path=None):
    """Send message and image to LINE Notify"""
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"}

    data = {"message": message}
    files = {"imageFile": open(image_path, "rb")} if image_path else None

    response = requests.post(url, headers=headers, data=data, files=files)
    if response.status_code == 200:
        print("LINE Notify sent successfully!")
    else:
        print("Failed to send LINE Notify:", response.text)

# Load video
cap = cv2.VideoCapture('D:/Thadzy/KatunyouAI/Fall-Detection/fall.mp4')

# Load YOLO Model
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
with open(r'D:\Thadzy\KatunyouAI\Fall-Detection\classes.txt', 'r') as f:
    classnames = f.read().splitlines()

fall_detected = False  # Prevent multiple alerts

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Error: Could not read frame from video.")
        break  # Exit loop if no frame is captured

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Implement fall detection using the coordinates x1,y1,x2
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

            if threshold < 0 and not fall_detected:
                cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2)

                # Save the frame as an image
                image_path = "fall_detected.jpg"
                cv2.imwrite(image_path, frame)
                
                # Send alert to LINE
                send_line_notify("🚨 Fall detected!", image_path)

                fall_detected = True  # Prevent multiple alerts

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
