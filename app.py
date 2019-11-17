import os
import cv2
from flask import Flask, flash, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import numpy as np

import matplotlib.pyplot as plt

UPLOAD_FOLDER = '/mnt/d/BHacks2019'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print(type(file))
            filestr = file.read()
            # convert string data to numpy array
            data = np.frombuffer(filestr, np.uint8)

            img = cv2.imdecode(data, cv2.IMREAD_COLOR)

            reshaped_img = cv2.resize(img, (224, 224))
            reshaped_img = cv2.cvtColor(reshaped_img, cv2.COLOR_BGR2RGB)

            print(img.shape, reshaped_img.shape)
            # RETURN RESHAPED IMAGE!

            return Response({'type': 'yes'}, status=200, mimetype="application/json")
            
    return  Response({'type': 'yes'}, status=200, mimetype="application/json")

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if (__name__ == '__main__'):
    app.run('127.0.0.1', port=3000)
