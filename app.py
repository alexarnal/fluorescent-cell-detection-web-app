from flask import Flask, render_template, url_for, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from detection.detect import run
from datetime import datetime
import shutil
import glob
import os 

#os.chdir('Documents/web-app-fluorescent-cell-detection/') #uncomment and modify if you'd like to call the app.py file from a different folder

def predict(inFileName, channel):
    print(f"\n\n\n\nRunning Prediction on {inFileName}")
    run(weights='detection/best.pt', source=inFileName, project='', channel=channel)
    os.remove(inFileName)
    shutil.rmtree('exp') 
    for path in glob.glob("exp*"):
        shutil.rmtree(path)
    print("\n\n\n\nfinished predicting")

app= Flask(__name__)
app.config['IMAGE_UPLOADS'] = 'uploads' # folder where to load images and store program outputs
app.config['OUTPUT_NAME'] = 'output' # will be overwritten by every filename

@app.route('/', methods=['POST','GET']) #GET is default without this method parameter
def index():
    download = False # indicates if download of program outputs is possible
    multiple = False # indicates if user uploaded multiple files 
    for f in os.listdir(app.config['IMAGE_UPLOADS']):
        os.remove(os.path.join(app.config['IMAGE_UPLOADS'], f))
    if request.method == 'POST':
        if request.files:
            for image in request.files.getlist('image[]'): #this is where you get the python input with id content
                channel = request.form['channel']
                image.save(os.path.join(app.config['IMAGE_UPLOADS'],image.filename))
                app.config['OUTPUT_NAME'] = os.path.splitext(image.filename)[0] #assumes file names do not contain periods other than for extensions
                try:
                    inFileName = os.path.join(app.config['IMAGE_UPLOADS'],image.filename)
                    predict(inFileName, channel)
                    download=True
                except Exception as ex:
                    print('There was an issue running cell detection:', ex)
            if len(request.files.getlist('image[]'))>1:
                print('uploaded multiple')
                multiple=True
                shutil.make_archive('download', 'zip', 'uploads')
                return render_template('index.html', download=download, multiple=multiple)
            multiple=False
            return render_template('index.html', download=download, multiple=multiple)
    else:
        return render_template('index.html', download=download, multiple=multiple)

@app.route('/download_file')
def download_file():
    p = os.path.join('uploads',app.config['OUTPUT_NAME']+'.svg')
    return send_file(p,as_attachment=True)

@app.route('/download_zip')
def download_zip():
    p = os.path.join('download.zip')
    return send_file(p,as_attachment=True)

@app.route('/')
def again():
    download=False
    multiple=False
    return render_template('index.html', download=download, multiple=multiple)

import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open_new('http://127.0.0.1:3000/') 

if __name__ == '__main__':
    Timer(1,open_browser).start(); 
    app.run(port=3000)