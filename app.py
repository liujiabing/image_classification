import json

import numpy as np
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def image_classifier():
    if 'file' not in request.files:
        return jsonify({"msg": "no image file", "status": -1})
    file = request.files['file']
    #if file.filename == '':
    #    return {"msg": "no file name", "status": -1}
    if not file:
        return jsonify({"msg": "no image body", "status": -1})
    img = image.img_to_array(image.load_img(file, target_size=(128, 128))) / 255.

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/test:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    return jsonify(pred)
    # Returning JSON response to the frontend
