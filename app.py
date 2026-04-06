from flask import Flask, render_template, Response, send_from_directory, jsonify
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load videos
video1 = cv2.VideoCapture(os.path.join(app.root_path, "video1.mp4"))
video2 = cv2.VideoCapture(os.path.join(app.root_path, "video2.mp4"))

vehicle_classes = [2, 3, 5, 7]

lane1_count = 0
lane2_count = 0
current_signal = ""


def generate_frames():
    global lane1_count, lane2_count, current_signal

    while True:
        success1, frame1 = video1.read()
        success2, frame2 = video2.read()

        print("Video1:", success1, "Video2:", success2)  # 👈 ADD THIS
       
        if not success1:
            video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if not success2:
            video2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame1 = cv2.resize(frame1, (640, 360))
        frame2 = cv2.resize(frame2, (640, 360))

        results1 = model(frame1)
        results2 = model(frame2)

        count1 = 0
        count2 = 0

        for r in results1:
            for box in r.boxes:
                if int(box.cls[0]) in vehicle_classes:
                    count1 += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for r in results2:
            for box in r.boxes:
                if int(box.cls[0]) in vehicle_classes:
                    count2 += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)

        lane1_count = count1
        lane2_count = count2

        if count1 > count2:
            current_signal = "Lane 1 Green"
        elif count2 > count1:
            current_signal = "Lane 2 Green"
        else:
            current_signal = "Equal Traffic"

        combined = cv2.hconcat([frame1, frame2])

        ret, buffer = cv2.imencode('.jpg', combined)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 🔹 MAIN PAGE
@app.route('/')
def home():
    return render_template('index.html')


# 🔹 VIDEO STREAM
@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 🔹 LIVE DATA
@app.route('/data')
def data():
    return jsonify({
        "lane1": lane1_count,
        "lane2": lane2_count,
        "signal": current_signal
    })


# 🔥 SIMULATION PAGE
@app.route('/simulation')
def simulation():
    return send_from_directory(
        os.path.join(app.root_path, 'my_traffic'),
        'index.html'
    )


# 🔥 SERVE ALL FILES (CSS + MP4)
@app.route('/my_traffic/<path:filename>')
def serve_files(filename):
    return send_from_directory(
        os.path.join(app.root_path, 'my_traffic'),
        filename
    )


if __name__ == '__main__':
    app.run(debug=True)