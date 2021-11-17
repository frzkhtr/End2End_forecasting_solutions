from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import cv2
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_resnet.h5'

detection_model = load_model('detection_model (1).h5')
location_model = load_model('location_prediction (2).h5')
severity_model = load_model('severity_prediction (2).h5')
img_size = 256

def damage_text(l):
  a = []
  for i in range(0, len(l)):
    if l[i] == 1:
      a.append('Damaged')
    else:
      a.append('Not_damaged')
  return a

def location_text(l):
  a = []
  for i in range(0, len(l)):
    if l[i] == 0:
      a.append('Front')
    elif l[i] == 1:
      a.append('Rear')
    else:
      a.append('Side')
  return a

def severity_text(l):
  a = []
  for i in range(0, len(l)):
    if l[i] == 0:
      a.append('Minor')
    elif l[i] == 1:
      a.append('Moderate')
    else:
      a.append('Severe')
  return a



def predict_damage(img_path):
    data = cv2.imread(img_path)[...,::-1]
    data = cv2.resize(data, (img_size, img_size))
    damage_predict = np.argmax(detection_model.predict(np.array([data])), axis = 1)
    damage_predict = damage_text(damage_predict)
    location_predict = np.argmax(location_model.predict(np.array([data])), axis = 1)
    location_predict = location_text(location_predict)
    severity_predict = np.argmax(severity_model.predict(np.array([data])), axis = 1)
    severity_predict = severity_text(severity_predict)
    return (severity_predict[0] + ' ' + location_predict[0] + ' ' + damage_predict[0])



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_damage(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)