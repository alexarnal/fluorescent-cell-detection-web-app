from flask import Flask, render_template, url_for, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os 

'''import re
import sys
import base64
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal, gaussian_kde
from matplotlib import cm
from matplotlib.colors import ListedColormap
from glob import glob'''

#slice
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys
import cv2
#import os

def slice(fileName):
    print("\nSet Up Slices Folder")    
    dataDir = ''
    processedDir = dataDir+"slices/"
    try:
        shutil.rmtree(processedDir)
    except:
        print(processedDir, "does not exist")
    os.mkdir(processedDir)

    print("\nLoading Data") 
    print(fileName)
    imageName = fileName
    temp = cv2.imread(dataDir+imageName)[:,:,0] #green channel
    print(type(temp), temp.dtype, np.max(temp), temp.shape)
    image = np.zeros((temp.shape[0],temp.shape[1],3))
    image[:,:,0]=temp
    image[:,:,1]=temp
    image[:,:,2]=temp

    print("\nSplitting & Saving Data\n\n")   
    frameSize = 256
    #cellSize = 36
    stride = frameSize
    indx = 0
    for row in range(0, image.shape[0]-frameSize, stride):
        for col in range(0, image.shape[1]-frameSize, stride):
            #print(row,col)
            frm = np.array([row,row+frameSize,col,col+frameSize]).astype('int')
            #print(frm)
            imageName = processedDir + "%s"%indx + '.png'
            print(f"Saving {imageName}: {image[frm[0]:frm[1],frm[2]:frm[3]].shape}")
            cv2.imwrite(imageName, image[frm[0]:frm[1],frm[2]:frm[3]])  #plt.imsave(processedDir + imageName + '.png',image[frm[0]:frm[1],frm[2]:frm[3]]) #, cmap='gray')
            indx+=1
            #print(indx)



#detect
from detection.detect import run

#stitch
#import matplotlib.pyplot as plt
#import numpy as np
#import shutil
#import sys
#import os
import re

def getYOLODataFromTXT(fileName,scale):
    file = open(fileName,'r')
    fileContent = file.readlines()
    file.close()
    #fileContent = fileContent.split('\n')
    cl,x,y,w,h,conf = [],[],[],[],[],[]
    for i,line in enumerate(fileContent): 
        if line == "": continue
        #print(line.split(' '))
        cl.append(int(line.split(' ')[0]))
        x.append(float(line.split(' ')[1])*scale)
        y.append(float(line.split(' ')[2])*scale)
        w.append(float(line.split(' ')[3])*scale)
        h.append(float(line.split(' ')[4])*scale)
        conf.append(float(line.split(' ')[5]))
    return np.array(cl), np.array(x), np.array(y), np.array(w), np.array(h), np.array(conf)

def startSVG(fileName,dims):
    f = open(fileName+".svg", "w")
    f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %s %s">\n'%(dims[0], dims[1]))
    f.write('<rect width="%s" height="%s" style="fill:none"/>\n'%(dims[0], dims[1]))
    f.close()

def saveCoordsSVG(x,y,fileName,indx): 
    f = open(fileName+".svg", "a")
    for i in range(x.shape[0]):
        f.write('<circle cx="%s" cy="%s" r="%s" style="fill:#000000"/>\n'%(x[i], y[i], 3))
    f.close()

def endSVG(fileName):
    f = open(fileName+".svg", "a")
    f.write('</svg>')
    f.close()    

def stitch(inFileName):  
    dataDir = ''
    processedDir = dataDir+"stitched/"
    try:
        shutil.rmtree(processedDir)
    except:
        print(processedDir, "does not exist")
    os.mkdir(processedDir)

    print("\nLoading Data") 
    print(dataDir+inFileName)
    #imageName = inFileName.copy()
    temp = cv2.imread(dataDir+inFileName)[:,:,1] #green channel
    print(type(temp), temp.dtype, np.max(temp), temp.shape)
    image = np.zeros((temp.shape[0],temp.shape[1],3))
    image[:,:,0]=temp
    image[:,:,1]=temp
    image[:,:,2]=temp
    #canvas = np.zeros((temp.shape[0],temp.shape[1],3))
    
    #imageFolder = 'images' #listdir(dataDir)
    labelFolder = 'exp/labels/'

    #onlyImgFiles = [f for f in os.listdir(dataDir+imageFolder) if os.path.isfile(os.path.join(dataDir+imageFolder, f))]
    #onlyImgFiles.sort(key=lambda f: int(re.sub('\D', '', f)))

    onlyLblFiles = [f for f in os.listdir(dataDir+labelFolder) if os.path.isfile(os.path.join(dataDir+labelFolder, f))]
    onlyLblFiles.sort(key=lambda f: int(re.sub('\D', '', f)))

    print("\nStitching based on reference image")   
    frameSize = 256
    #cellSize = 36
    stride = frameSize
    indx = 0
    imageDims = [image.shape[1],image.shape[0]]
    startSVG(processedDir + "stitched",imageDims)
    for row in range(0, image.shape[0]-frameSize, stride):
        for col in range(0, image.shape[1]-frameSize, stride):
            if os.path.isfile(dataDir+labelFolder+str(indx)+".txt") == False: 
                indx+=1
                continue
            #frm = np.array([row,row+frameSize,col,col+frameSize]).astype('int')
            #labelName = "labels/train/%s"%trainCount
            #canvas[frm[0]:frm[1],frm[2]:frm[3]] += plt.imread(dataDir+'images/'+str(indx)+".png")
            #trainCount+=1
            c,x,y,w,h,conf=getYOLODataFromTXT(dataDir+labelFolder+str(indx)+".txt",frameSize)
            
            #account for relative position of patch in large image
            x += col
            y += row
            #saveCoordsTXT(c, x, y, w, h, processedDir + "stitched",indx)
            saveCoordsSVG(x, y, processedDir + "stitched",indx)
            indx+=1
        #if valCount == 10: break
    endSVG(processedDir + "stitched")

    print("Done!")

#main
def predict(inFileName):
    print("\n\n\n\ninside prediction")
    slice(inFileName)
    print("\n\n\n\nimage has been split")
    run(weights='detection/best.pt', source="slices/", imgsz=(256,256), save_txt=True, save_conf=True, project='')
    print("\n\n\n\nfinished predicting")
    stitch(inFileName)
    print("\n\n\n\nfinished stitching predictions")
    

app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db' #4/is absolute path
app.config['IMAGE_UPLOADS'] = 'uploads' #4/is absolute path
#MYDIR = os.path.dirname(__file__)
#print("\n\n\n"+MYDIR+"this one <-----\n\n\n")

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
    for f in os.listdir(app.config['IMAGE_UPLOADS']):
        #if not f.endswith(".bak"):
        #    continue
        os.remove(os.path.join(app.config['IMAGE_UPLOADS'], f))
    if request.method == 'POST':
        if request.files:
            image = request.files['image'] #this is where you get the python input with id content
            image.save(os.path.join(app.config['IMAGE_UPLOADS'],image.filename))#svg.filename))
            
            try:
                inFileName = os.path.join(app.config['IMAGE_UPLOADS'],image.filename)
                #outFileName = os.path.join(app.config['IMAGE_UPLOADS'],'output.svg')
                predict(inFileName)
                #return redirect(request.url)
                download=True
            except:
                print('There was an issue running cell detection')
            return render_template('index.html', download=download)#redirect(request.url) 
        #new_task = ToDo(content=fileName)
        
        #try:
        #    db.session.add(svg.filename)
        #    db.session.commit()
        #    return redirect('/')
        #except:
        #    return "There was an issue adding your task"
        
    else:
        #tasks = ToDo.query.order_by(ToDo.date_created).all()
        
        return render_template('index.html', download=download)#tasks=tasks, )
    
@app.route('/download')
def download_file():
    p = os.path.join('stitched','stitched.svg')
    return send_file(p,as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    