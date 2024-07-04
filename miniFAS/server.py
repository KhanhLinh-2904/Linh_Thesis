import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from function_model.fas import FaceAntiSpoofing
import numpy as np
import cv2
import os
import multiprocessing as mp
import psutil

app = Flask(__name__)
CORS(app)

# Paths to models
fas1_lowlight_path = "model_onnx/train_SCI_miniFAS/2.7_80x80_MiniFASNetV2.onnx"
fas2_lowlight_path = "model_onnx/train_SCI_miniFAS/4_0_0_80x80_MiniFASNetV1SE.onnx"
fas1_normal_path = "model_onnx/2.7_80x80_MiniFASNetV2.onnx"
fas2_normal_path = "model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"

# Initialize models
fas1_lowlight = FaceAntiSpoofing(fas1_lowlight_path)
fas2_lowlight = FaceAntiSpoofing(fas2_lowlight_path)
fas1_normal = FaceAntiSpoofing(fas1_normal_path)
fas2_normal = FaceAntiSpoofing(fas2_normal_path)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / (1024 * 1024):.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB")

def anti_spoofing_lowlight(img, return_dict):
    pred1 = fas1_lowlight.predict(img)
    pred2 = fas2_lowlight.predict(img)
    if pred1 is None or pred2 is None:
        return_dict['result'] = None
    else:
        prediction = pred1 + pred2
        output = np.argmax(prediction)
        return_dict['result'] = output

def anti_spoofing_normal(img, return_dict):
    pred1 = fas1_normal.predict(img)
    pred2 = fas2_normal.predict(img)
    if pred1 is None or pred2 is None:
        return_dict['result'] = None
    else:
        prediction = pred1 + pred2
        output = np.argmax(prediction)
        return_dict['result'] = output

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'frame' not in request.files or 'message' not in request.form:
        return jsonify({'message': 'Missing data'}), 400
    start_time = time.time()
    frame = request.files['frame']
    
    message = request.form['message']
    file_bytes = np.frombuffer(frame.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    print(f"Received message: {message}")
    
    log_memory_usage()  # Log memory usage before processing

    manager = mp.Manager()
    return_dict = manager.dict()
    
    if message == 'enhance':
        p = mp.Process(target=anti_spoofing_lowlight, args=(image, return_dict))
    else:
        p = mp.Process(target=anti_spoofing_normal, args=(image, return_dict))
    
    p.start()
    p.join()

    output = return_dict.get('result', None)
    end_time = time.time()
    print("Time for 1 request: ", end_time-start_time)
    
    log_memory_usage()  # Log memory usage after processing

    if output is None:
        return jsonify({"message": "No Face"}), 200
    elif output == 1:
        return jsonify({"message": "Real"}), 200
    else:
        return jsonify({"message": "Fake"}), 200

if __name__ == '__main__':
    app.run(debug=True)