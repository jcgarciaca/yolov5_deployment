import cv2
import requests
import os
import json
import sys
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

"""
Usage:
python run_inference.py source_image_path

Example:
python run_inference.py /home/plate-recognition/Downloads/jmf_car/images_sample/3.jpg
"""

def build_tesseract_options(psm=7):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    # options += " --psm {} --oem 3".format(psm)
    options += " --psm {}".format(psm)
    # return the built options string
    return options

options = build_tesseract_options(psm=7)

URL = 'http://192.168.56.122:5080/inference'

img_path = sys.argv[1]
img_name = os.path.basename(img_path)
root_folder = os.path.dirname(img_path)
roi_folder = os.path.join(root_folder, 'roi')

print('Running inference for {}'.format(img_name))
colors = {'plate': (0, 0, 255)}

try:
    files = {'img': open(img_path, 'rb')}
    r = requests.post(URL, files=files)
    resp = r.json()
    
    img = Image.open(img_path)
    for _, detection in resp.items():
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(font='/usr/share/vlc/skins2/fonts/FreeSans.ttf', size=20)
        xmin = round(detection['xmin'])
        ymin = round(detection['ymin'])
        xmax = round(detection['xmax'])
        ymax = round(detection['ymax'])            
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=colors[detection['name']], width=3)
        # draw.text((xmin + 5, ymin + 5), detection['name'], colors[detection['name']], font=font)
        # roi = np.array(img.crop((xmin, ymin, xmax, ymax)))[:, :, ::-1] #.convert('L'))
        # roi_name = '{}.jpg'.format(len(os.listdir(roi_folder)) + 1)
        # cv2.imwrite(os.path.join(roi_folder, roi_name), roi)
        # roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # print('text:', pytesseract.image_to_string(roi, config=options))
    output = np.array(img)[:, :, ::-1]
    cv2.imshow('output', output)
    cv2.waitKey(0)
        
except Exception as ex:
    print('Error with running inference on image {} Exception: {}'.format(img_name, ex))
