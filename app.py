import cv2
import numpy as np
from flask import Flask, render_template, Response, request, send_file
import io
import time
import os

# --- Pre-flight Check: Verify model files exist before starting ---
required_files = ["yolov4-tiny.weights", "yolov4-tiny.cfg", "coco.names"]
for f in required_files:
    if not os.path.exists(f):
        print(f"[FATAL ERROR] The required file '{f}' was not found.")
        print("Please make sure all model files are in the same directory as app.py.")
        exit()
# -----------------------------------------------------------------

# Initialize the Flask app
app = Flask(__name__)

# --- Model Loading (Using YOLOv4-tiny) ---
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

try:
    output_layers_names = net.getUnconnectedOutLayersNames()
except AttributeError:
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers_names = [layer_names[i[0] - 1] for i in output_layers_indices]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
# ------------------------------------

def enhance_image(image):
    """Enhances the clarity of an image for better object detection."""
    # Convert to LAB color space to work on luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Apply a gentle denoising
    final_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 10, 10, 7, 21)
    
    return final_image

def detect_objects(frame, confidence_threshold=0.5):
    """
    Runs object detection with a configurable confidence threshold.
    Returns the processed frame and object count.
    """
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers_names)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Use the provided confidence threshold
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    
    object_count = 0
    if len(indexes) > 0:
        object_count = len(indexes)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            text = f"{label}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5), font, 1.5, color, 2)
            
    return frame, object_count

def generate_frames():
    """Generator for the webcam video stream."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
        
    prev_frame_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Live video uses the default, stricter confidence for better performance
        frame, object_count = detect_objects(frame) 
        
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Objects: {object_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return "No image file provided", 400
    file = request.files['image']
    if file.filename == '':
        return "No image selected", 400
        
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if image is None:
        return "Invalid image file", 400
    
    # --- NEW: Enhance the uploaded image before detection ---
    enhanced_image = enhance_image(image)
    
    # Use the enhanced image and a lower confidence threshold for better results on unclear photos
    processed_image, object_count = detect_objects(enhanced_image, confidence_threshold=0.4)
    
    cv2.putText(processed_image, f"Objects Detected: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    ret, buffer = cv2.imencode('.jpg', processed_image)
    if not ret:
        return "Failed to encode image", 500
        
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)