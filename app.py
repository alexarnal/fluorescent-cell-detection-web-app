from flask import Flask, render_template, url_for, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os 

import re
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
from glob import glob

def newColorMap(oldCmap, nColors, reverse=False, opacity=True):
    #https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
    big = cm.get_cmap(oldCmap, nColors) #sample a large set of colors from cmap
    if reverse: newCmap = big(np.linspace(1, 0, nColors))
    else: newCmap = big(np.linspace(0, 1, nColors))
    if opacity: newCmap[:,3] = np.linspace(0, 1, nColors)
    return ListedColormap(newCmap)
  
def getCoordsFromSVG(fileName):
    file = open(fileName,'r')
    fileContent = file.readline()
    file.close()
    fileContent = fileContent.split('>')
    X=[]
    Y=[]
    viewBox = 0
    for i,line in enumerate(fileContent): 
        coord = 0
        if 'viewBox=' in line:
            viewBox = re.search('viewBox="(.*)"', line)
        if 'circle class=' in line:
            x = re.search('cx="(.*)" cy', line)
            y = re.search('cy="(.*)" r', line)
            coord = [x.group(1),y.group(1)]
        elif 'path class=' in line:
            coord = re.search('"M(.*?)a', line, re.I)
            coord = coord.group(1).split(',')   
        else: continue
        X.append(float(coord[0]))
        Y.append(-float(coord[1]))
    coords = np.vstack((np.array(X), np.array(Y))).T
    return coords, np.array(viewBox[1].split(' '), dtype='float')

def cellDensityMap3(coords,viewBox, sigma):
    #first remove duplicates
    coords = [tuple(row) for row in coords]
    coords = np.unique(coords,axis=0)
    
    x=-coords[:,1]
    y=coords[:,0]
    im = np.zeros((int(viewBox[3]),int(viewBox[2])))
    for i in range(len(x)):
        try:
            # print(int(x[i]),int(y[i]))
            im[int(x[i]),int(y[i])] += 1
        except: 
            print('Cell out of frame - Skipping')
            continue
    return gaussian_filter(im,sigma)

def contour1(im, coords, filename, vmin, vmax, cmap):
    #Determine where to draw contours (levels) - 1%, 5%, 10%, 20%, 40%, 60%,
    #   80% (and 100% when the density's maximum value is equal to vmax: a
    #   work around matplotlib.pyplot.contourf()'s 'color fill' and 'levels'
    #   mapping). 
    #may not need this - lvls = list(np.linspace(np.min(im),np.max(im),6))[1:] #increments of 20%
    densityRange = np.max(im)-np.min(im)
    lvls = np.min(im) + (densityRange*np.array((0.01,0.05,0.1,0.2,0.4,
                                                0.6,0.8,0.9,0.95,1.0)))
    if np.max(im)==vmax: lvls=lvls[:-1] #remove highest level 
    
    # may not need this - lvls = [np.min(im)+densityRange*0.01, np.min(im)+densityRange*0.05,
    #         np.min(im)+densityRange*0.10] + lvls
    
    #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf
    x = np.linspace(0, im.shape[1], im.shape[1])
    y = np.linspace(0, im.shape[0], im.shape[0])
    X, Y = np.meshgrid(x,y)
    print("\n\n\n\ngot XY")
    print("\n\n\n\n%s"%filename)
    fig = Figure()
    ax = fig.subplots()
    ax.scatter(coords[:,0],coords[:,1])
    ax.contourf(X, -Y, im, levels=lvls, cmap=cmap,origin='image', vmax=vmax, vmin=vmin, extend='both')
    ax.axis('off')
    
    # Save it to a temporary buffer.
    fig.savefig(filename, format="svg")

    #plt.contourf(X, -Y, im, levels=lvls, cmap=cmap, 
    #             origin='image', vmax=vmax, vmin=vmin, extend='both')
    #print("\n\n\n\ngot contour")
    #plt.axis('off')
    #plt.savefig(filename)
    #plt.clf() 
    #print("\n\n\n\ngot contour")

def newColorMap(oldCmap, nColors, reverse=False, opacity=True):
    #https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
    big = cm.get_cmap(oldCmap, nColors) #sample a large set of colors from cmap
    if reverse: newCmap = big(np.linspace(1, 0, nColors))
    else: newCmap = big(np.linspace(0, 1, nColors))
    if opacity: newCmap[:,3] = np.linspace(0, 1, nColors)
    return ListedColormap(newCmap)

def estimateDensity(inFileName, outFileName, sigma):
    print("\n\n\n\ninside estimation")
    coords, viewBox = getCoordsFromSVG(inFileName)
    print("\n\n\n\ngot coords")
    density = cellDensityMap3(coords,viewBox,sigma)
    print("\n\n\n\ngot density")
    contour1(density, coords, outFileName, vmin = None, vmax = None, cmap = newColorMap('viridis', 1000, opacity=True, reverse=False))
    
    print("\n\n\n\ngot contour")
    sys.stdout.flush()

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
            svg = request.files['SVG'] #this is where you get the python input with id content
            svg.save(os.path.join(app.config['IMAGE_UPLOADS'],svg.filename))#svg.filename))
            
            try:
                inFileName = os.path.join(app.config['IMAGE_UPLOADS'],svg.filename)
                outFileName = os.path.join(app.config['IMAGE_UPLOADS'],'output.svg')
                estimateDensity(inFileName, outFileName, 2.5)
                #return redirect(request.url)
                download=True
            except:
                print('There was an issue estimating density')
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
    p = os.path.join(app.config['IMAGE_UPLOADS'],'output.svg')
    return send_file(p,as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)