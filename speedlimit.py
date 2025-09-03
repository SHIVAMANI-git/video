import cv2
import csv
import time
import math
from ultralytics import YOLO
from collections import defaultdict

# -------------------------------
# Ask user for source
# -------------------------------
print("Select video source:")
print("1. Webcam")
print("2. Video file")
choice = input("Enter choice (1/2): ")

if choice == "1":
    video_source = 0  # Webcam
else:
    video_source = input("Enter video file path (e.g. traffic.mp4): ")

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO("yolov8n.pt")  # use yolov8l.pt for better accuracy

cap = cv2.VideoCapture(video_source)

# -------------------------------
# Output CSV
# -------------------------------
csv_file = "detections.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "ObjectID", "Label", "X", "Y", "Speed (km/h)", "Alert"])

# -------------------------------
# Variables for speed calculation
# -------------------------------
object_positions = defaultdict(list)
PIXEL_TO_METER = 0.05  # adjust for your camera setup
fps = cap.get(cv2.CAP_PROP_FPS) or 30
SPEED_LIMIT = 80  # km/h

# -------------------------------
# Processing Loop
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_id = int(box.id[0]) if box.id is not None else None

            if obj_id is not None:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                object_positions[obj_id].append((time.time(), cx, cy))

                speed = 0
                alert = "No"
                if len(object_positions[obj_id]) >= 2:
                    (t1, x1_old, y1_old), (t2, x2_new, y2_new) = object_positions[obj_id][-2:]
                    dist_pixels = math.sqrt((x2_new - x1_old) ** 2 + (y2_new - y1_old) ** 2)
                    dist_meters = dist_pixels * PIXEL_TO_METER
                    time_elapsed = t2 - t1
                    if time_elapsed > 0:
                        speed = (dist_meters / time_elapsed) * 3.6

                color = (0, 255, 0)
                if speed > SPEED_LIMIT and label in ["car", "bus", "truck", "motorbike"]:
                    color = (0, 0, 255)
                    alert = "Overspeeding"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {speed:.1f} km/h",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if alert == "Overspeeding":
                    cv2.putText(frame, "🚨 OVERSPEED 🚨",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                     obj_id, label, cx, cy, f"{speed:.2f}", alert])

    cv2.imshow("YOLO Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
