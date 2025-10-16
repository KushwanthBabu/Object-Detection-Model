import cv2
import numpy as np
from flask import Flask, render_template, Response, request, send_file
import io

# Initialize the Flask app
app = Flask(__name__)

# --- Model Loading (same as before) ---
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]
try:
    output_layers = net.getUnconnectedOutLayersNames()
except AttributeError:
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# ------------------------------------

def detect_objects(frame):
    """Runs object detection on a single frame or image."""
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf > 0.5:
                cx, cy, w, h = (det[0] * width, det[1] * height, det[2] * width, det[3] * height)
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(conf)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), font, 1.5, color, 2)
    return frame

def generate_frames():
    """Generator for the webcam video stream."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection on the frame
        frame = detect_objects(frame)
        
        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provides the video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Handles image upload and returns the processed image."""
    if 'image' not in request.files:
        return "No image file provided", 400
    
    file = request.files['image']
    
    if file.filename == '':
        return "No image selected", 400

    # Read the image file into an OpenCV object
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        return "Invalid image file", 400

    # Run detection
    processed_image = detect_objects(image)

    # Encode the processed image to JPEG
    ret, buffer = cv2.imencode('.jpg', processed_image)
    if not ret:
        return "Failed to encode image", 500

    # Return the image file as a response
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)