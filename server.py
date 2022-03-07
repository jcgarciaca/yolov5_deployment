import json
import os
import torch
from flask import Flask, request, Response
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/JulioCesar/yolov5/runs/train/exp16/weights/best.pt')
#model.conf = 0.06
#print('conf', model.conf)
#print('iou', model.iou)

app = Flask(__name__)
@app.route('/inference', methods=['POST'])
def run_inference():
    img = Image.open(request.files['img'])
    results = model(img, size=1024)
    df = results.pandas().xyxy[0]
    json_data = df.to_json(orient='index')
    return Response(json_data, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5080', debug=True)
