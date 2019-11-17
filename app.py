import os
import cv2
from flask import Flask, flash, request, redirect, url_for, Response
from classifier.eval import evaluate

from werkzeug.utils import secure_filename
import numpy as np

import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
 
        if True in [file.endswith(extension) for extension in ALLOWED_EXTENSIONS]
            filestr = file.read()
            data = np.frombuffer(filestr, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            reshaped_img = cv2.cvtColor(reshaped_img, cv2.COLOR_BGR2RGB)


            return Response({'type': 'yes'}, status=200, mimetype="application/json")
            
    return  Response({'type': 'yes'}, status=200, mimetype="application/json")

if (__name__ == '__main__'):
    app.run('127.0.0.1', port=3000)
