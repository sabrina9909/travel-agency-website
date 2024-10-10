from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Load models
models = {
    'drones': YOLO(r'D:\project\drones.pt'),
    'helmets': YOLO(r'D:\project\helmets.pt'),
    'masks': YOLO(r'D:\project\masks.pt'),
    'fire': YOLO(r'D:\project\fire.pt'),
    'backpacks': YOLO(r'D:\project\backpacks.pt'),
    'cars': YOLO(r'D:\project\cars.pt'),
    'guns': YOLO(r'D:\project\guns.pt'),
    'knives': YOLO(r'D:\project\knives.pt'),
    'motorcycles': YOLO(r'D:\project\motorcycles.pt'),
    'people': YOLO(r'D:\project\people.pt')
}

# Define label maps for each model
label_maps = {
    'drones': {0: 'drone', 1: 'not drone'},
    'helmets': {0: 'helmet', 1: 'no helmet'},
    'masks': {0: 'mask', 1: 'no mask'},
    'fire': {0: 'Fire', 1: 'Non-fire'},
    'backpacks': {0: 'backpack', 1: 'not backpack'},
    'cars': {0: 'car', 1: 'not car'},
    'guns': {0: 'gun', 1: 'not gun'},
    'knives': {0: 'knife', 1: 'not knife'},
    'motorcycles': {0: 'motorcycle', 1: 'not motorcycle'},
    'people': {0: 'person', 1: 'not person'}
}

# Positive labels
positive_labels = {
    'drones': 'drone',
    'helmets': 'helmet',
    'masks': 'mask',
    'fire': 'Fire',
    'backpacks': 'backpack',
    'cars': 'car',
    'guns': 'gun',
    'knives': 'knife',
    'motorcycles': 'motorcycle',
    'people': 'person'
}

# Assume 'fire' as default model
selected_model_name = 'fire'
selected_model = models[selected_model_name]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    global selected_model_name, selected_model
    data = request.json
    model_name = data.get('model')
    if model_name in models:
        selected_model_name = model_name
        selected_model = models[model_name]
        return jsonify({"status": "success", "selected_model": model_name})
    return jsonify({"status": "error", "message": "Model not found"}), 400

@app.route('/detect', methods=['POST'])
def detect():
    # Get the image from the request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform detection
    results = selected_model(img)

    # Format the results
    detections = []
    label_map = label_maps[selected_model_name]
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf.item())
            if confidence > 0.5:  # Only consider detections with confidence > 50%
                cls_id = int(box.cls.item())
                label = label_map.get(cls_id, f"Unknown({cls_id})")  # Use default if not found
                detection = {
                    'label': label,
                    'confidence': confidence,  # Convert tensor to float
                    'box': [float(coord) for coord in box.xyxy[0].tolist()]  # Ensure each coord is converted to float
                }
                if label == positive_labels[selected_model_name]:
                    detections.append(detection)
                    # Emit a socket event for each detection
                    emit_detection(detection)

    return jsonify(detections)

def emit_detection(detection):
    label = detection['label']
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Check if this detection has already been notified within the last minute
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('SELECT * FROM detections WHERE label=? AND time=?', (label, current_time))
    existing_detection = c.fetchone()
    
    if not existing_detection:
        c.execute('INSERT INTO detections (label, time) VALUES (?, ?)', (label, current_time))
        conn.commit()
        socketio.emit('new_detection', {'label': label, 'time': current_time})
    
    conn.close()

if __name__ == '__main__':
    socketio.run(app, debug=True)
