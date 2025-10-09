from flask import Flask, render_template, request
from threading import Thread
from ultralytics import YOLO
import cv2
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

violator_folder = "violators"
os.makedirs(violator_folder, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start_traffic():
    lanes = int(request.form['lanes'])
    video_sources = []

    # Save uploaded files
    for i in range(1, lanes + 1):
        file = request.files.get(f'lane{i}')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            video_sources.append(filepath)

    # Start traffic controller in a separate thread
    thread = Thread(target=run_traffic_controller, args=(video_sources,))
    thread.start()
    return f"Traffic Controller Started for {lanes} lanes."


def run_traffic_controller(video_sources):
    lane_count = len(video_sources)
    model = YOLO("yolov8n.pt")
    CONF_THRESH = 0.4
    VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]
    AMBULANCE_CLASS = "truck"

    lane_signals = ["RED"] * lane_count

    caps = [cv2.VideoCapture(src) for src in video_sources]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"❌ Could not open video source {video_sources[i]}")
            return

    while True:
        frames = []
        lane_status = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frames.append(None)
                lane_status.append("No Signal")
                continue

            results = model(frame, verbose=False)
            vehicle_detected = False
            ambulance_detected = False
            violator_detected = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    label = model.names[cls_id]
                    conf = float(box.conf.item())
                    if conf < CONF_THRESH:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Ambulance detection
                    if label == AMBULANCE_CLASS:
                        ambulance_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Ambulance", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Vehicle detection
                    if label in VEHICLE_CLASSES:
                        vehicle_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Lane violation detection (red signal)
                        if lane_signals[i] == "RED":
                            violator_detected = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "VIOLATOR", (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            timestamp = int(time.time() * 1000)
                            filename = os.path.join(violator_folder, f"violator_lane{i+1}_{timestamp}.jpg")
                            cv2.imwrite(filename, frame)
                            print(f"🚨 Violation detected! Saved: {filename}")

            if ambulance_detected:
                lane_status.append("AMBULANCE")
            elif vehicle_detected:
                lane_status.append("Occupied")
            else:
                lane_status.append("Empty")

            frames.append(frame)

        # Signal logic
        for i in range(lane_count):
            if lane_status[i] == "AMBULANCE":
                lane_signals[i] = "GREEN"
            else:
                lane_signals[i] = "GREEN" if lane_status[i] == "Occupied" else "RED"

        # Display frames
        for i, frame in enumerate(frames):
            if frame is not None:
                cv2.putText(frame, f"Signal: {lane_signals[i]}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                cv2.imshow(f"Lane {i + 1}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(debug=True)
