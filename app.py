import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

model = YOLO("yolov8n.pt")
video_path = "Battle machines in large convoy 🪖 🇬🇧 [Uh8ixVdSctM].mp4"
cap = cv2.VideoCapture(video_path)
tracker = DeepSort(max_age=30)

cv2.namedWindow("Tank Tracking", cv2.WINDOW_NORMAL)  # pencere boyutu ayarlanabilir


track_history = {}

def draw_reticle(frame):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    size = 40
    color = (0, 255, 0)
    thickness = 2
    cv2.line(frame, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness)
    cv2.line(frame, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness)
    cv2.circle(frame, center, 5, color, -1)

def draw_distance(frame, x1, y1, x2, y2):
    box_height = y2 - y1
    distance = int(1000 / (box_height + 1))
    cv2.putText(frame, f"Dist: {distance}m", (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_direction_arrow(frame, track_id, prev_pos, curr_pos):
    if prev_pos is None:
        return
    x_prev, y_prev = prev_pos
    x_curr, y_curr = curr_pos
    color = (0, 255, 255)
    thickness = 2
    cv2.arrowedLine(frame, (x_prev, y_prev), (x_curr, y_curr), color, thickness, tipLength=0.3)
    dx = x_curr - x_prev
    dy = y_curr - y_prev
    speed = int(np.sqrt(dx*dx + dy*dy))
    cv2.putText(frame, f"ID {track_id} Spd:{speed}", (x_curr, y_curr + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=max(frame_width, frame_height))[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if conf > 0.4:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        draw_distance(frame, x1, y1, x2, y2)

        prev_pos = track_history.get(track_id)
        curr_pos = (cx, cy)
        draw_direction_arrow(frame, track_id, prev_pos, curr_pos)

        track_history[track_id] = curr_pos

    draw_reticle(frame)

    cv2.imshow("Vehicle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
