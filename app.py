from flask import Flask, render_template, url_for, request, redirect, send_file
from segmentation.unet_predict import run
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import shutil
import os 

os.chdir('Documents/web-app-fluorescent-fiber-segmentation/')

def predict(inFileName, channel):
    print(f"\n\n\n\nRunning Prediction on {inFileName}")
    run(img_path=inFileName, channel=channel)
    os.remove(inFileName)
    print("\n\n\n\nfinished predicting")

app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['IMAGE_UPLOADS'] = 'uploads'
app.config['OUTPUT_NAME'] = 'output'
db = SQLAlchemy(app)

class ToDo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200),nullable=False)
    date_created = db.Column(db.DateTime, default = datetime.utcnow)

    def __repr__(self):
        return '<Tsk %r>' %self.id

@app.route('/', methods=['POST','GET']) #GET is default without this method parameter
def index():
    download=False
    multiple = False
    for f in os.listdir(app.config['IMAGE_UPLOADS']):
        os.remove(os.path.join(app.config['IMAGE_UPLOADS'], f))
    if request.method == 'POST':
        if request.files:
            for image in request.files.getlist('image[]'): #this is where you get the python input with id content
                channel = request.form['channel']
                image.save(os.path.join(app.config['IMAGE_UPLOADS'],image.filename))
                app.config['OUTPUT_NAME'] = os.path.splitext(image.filename)[0]
                try:
                    inFileName = os.path.join(app.config['IMAGE_UPLOADS'],image.filename)
                    predict(inFileName, channel)
                except Exception as ex:
                    print('There was an issue running cell detection:', ex)
            download=True #switch to allow for download page to appear
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
    webbrowser.open_new('http://127.0.0.1:2000/') 

if __name__ == '__main__':
    Timer(1,open_browser).start(); 
    app.run(port=2000)