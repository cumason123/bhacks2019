import os
import cv2
from flask import Flask, flash, request, redirect, url_for, Response
from classifier.eval import evaluate

from werkzeug.utils import secure_filename
import numpy as np

import matplotlib.pyplot as plt
from classifier.model import TransferModel

skin_classifier = TransferModel(model_cls, len(test_dataset.classes))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if True in [file.filename.endswith(extension) for extension in ALLOWED_EXTENSIONS]:
            filestr = file.read()
            data = np.frombuffer(filestr, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            reshaped_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resp = evaluate(np.array(reshaped_img))

            return Response({'type': resp}, status=200, mimetype="application/json")
    return  Response([], status=200, mimetype="application/json")

if (__name__ == '__main__'):
    app.run('0.0.0.0', port=3000)
