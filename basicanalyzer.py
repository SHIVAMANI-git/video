import cv2
import csv
import time
from ultralytics import YOLO

# Load YOLOv8 nano model (small & fast)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# CSV file setup
csv_file = open("detections.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Time", "Object", "Confidence", "X", "Y", "Width", "Height"])

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Label + confidence
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save detection to CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([timestamp, label, conf, x1, y1, x2 - x1, y2 - y1])

    # Show video
    cv2.imshow("Webcam Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
