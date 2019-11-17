import os
import cv2
from flask import Flask, flash, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import numpy as np

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
            print(img.shape)
            # convert numpy array to image
            # img = cv2.imdecode(npimg, )

            return Response({"Hello":"hi"}, status=200, mimetype="application/json")
            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if (__name__ == '__main__'):
    app.run('127.0.0.1', port=3000)
