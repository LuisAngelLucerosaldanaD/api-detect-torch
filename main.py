import argparse
import io
import base64
import json
import shutil

from PIL import Image

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

DETECTION_URL = "/api/v1/prediction"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    file = request.files['img-encoding']
    if file.mimetype != "image/png" and file.mimetype != "image/jpeg" and file.mimetype != "application/octet-stream":
        return jsonify({'error': True,
                        'data': [],
                        'code': 22,
                        'type': 'error',
                        'msg': 'El archivo debe ser de formato imagen: jpeg, jpg, png'})

    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img, size=640)
    data = results.pandas().xyxy[0].to_json(orient="records")

    results.render()
    results.save()
    path_img = './runs/detect/exp'
    with open(path_img + '/image0.jpg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    shutil.rmtree(path_img)
    predictions_json = json.loads(data)
    return jsonify({'error': False,
                    'data': {
                        'predictions': predictions_json,
                        'file': encoded_image
                    },
                    'code': 29,
                    'type': 'success',
                    'msg': 'procesado correctamente'})


if __name__ == "__main__":

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt')
    model.eval()
    app.run(port=6020)
