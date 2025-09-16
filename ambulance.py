from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO("yolov8m.pt")  # small & fast model, use yolov8s.pt for better accuracy

# Load video file (or use 0 for webcam)
cap = cv2.VideoCapture("images/ambulance.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])         # class id
            label = model.names[cls_id]      # class name
            conf = float(box.conf[0])        # confidence score

            if label == "truck":  # 👈 Treat every truck as ambulance
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Ambulance", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("🚨 Ambulance detected!")

    cv2.imshow("Ambulance Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
