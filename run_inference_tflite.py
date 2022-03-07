import cv2
import time
import numpy as np
import tensorflow as tf


def classFilter(classdata):
     classes = []  # create a list
     for i in range(classdata.shape[0]):         # loop through all predictions
         classes.append(classdata[i].argmax())   # get the best classification location
     return classes  # return classes (int)


def YOLOdetect(output_data):
     output_data = output_data[0]
     boxes = np.squeeze(output_data[..., :4])
     scores = np.squeeze(output_data[..., 4:5])
     classes = classFilter(output_data[..., 5:])
     x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
     xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
     return xyxy, classes, scores


path = '/home/JulioCesar/yolov5/runs/train/exp17/weights/best-fp16.tflite'
img_path = '/home/JulioCesar/plate_detection/images_sample/3.jpg'

img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA).astype(np.float32)
height, widht, _ = img.shape
show_img = img.copy()
img /= 255.

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
start_time = time.time()

interpreter.set_tensor(input_details[0]['index'], img[np.newaxis, ...])

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()

xyxy, classes, scores = YOLOdetect(output_data) #boxes(x,y,x,y), classes(int), scores(float) [25200]

print(output_data)

print('Check every box')

cnt = 0
for i in range(len(scores)):
     if (scores[i] > 0.25):
          cnt += 1
          xmin = int(max(1, (xyxy[0][i] * widht)))
          ymin = int(max(1, (xyxy[1][i] * height)))
          xmax = int(min(height, (xyxy[2][i] * widht)))
          ymax = int(min(widht, (xyxy[3][i] * height)))
          img = cv2.rectangle(show_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

print('Found {} boxes'.format(cnt))
cv2.imwrite('/home/JulioCesar/plate_detection/images_sample/inference/tflite_3.jpg', show_img)

