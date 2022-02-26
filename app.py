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
import os

def slice(fileName):
    print("\nSet Up")    
    dataDir = ''
    processedDir = dataDir+"slices/"
    try:
        shutil.rmtree(processedDir)
    except:
        print(processedDir, "does not exist")
    os.mkdir(processedDir)

    print("\nLoading Data")   
    imageName = fileName.copy()
    temp = plt.imread(dataDir+imageName)#[:,:,0] #green channel
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
            frm = np.array([row,row+frameSize,col,col+frameSize]).astype('int')
            imageName = "%s"%indx
            plt.imsave(processedDir + imageName + '.png',image[frm[0]:frm[1],frm[2]:frm[3]], cmap='gray')
            indx+=1



#detect
#import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

#@torch.no_grad()
def detect():
    weights=ROOT / 'best.pt',  # model.pt path(s)
    source=ROOT / 'slices',  # file/dir/URL/glob, 0 for webcam
    #data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(256, 256),  # inference size (height, width)
    conf_thres=0.566,  # confidence threshold #obtained from f1 curve
    iou_thres=0.3,  # NMS IOU threshold #default is 0.45, we used 0.2 with very conservative results
    max_det=1000,  # maximum detections per image
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=True,  # save results to *.txt
    save_conf=True,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT ,  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=1,  # bounding box thickness (pixels)
    hide_labels=True,  # hide labels
    hide_conf=True,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference

    source = str(source)
    #print(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)#, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


#stitch
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys
import os
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
    imageName = inFileName.copy()
    temp = plt.imread(dataDir+imageName)[:,:,1] #green channel
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
            frm = np.array([row,row+frameSize,col,col+frameSize]).astype('int')
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
    detect()
    print("\n\n\n\nfinished predicting")
    stitch(inFileName)
    print("\n\n\n\nfinished stitching predictions")
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
            image = request.files['image'] #this is where you get the python input with id content
            image.save(os.path.join(app.config['IMAGE_UPLOADS'],image.filename))#svg.filename))
            
            try:
                inFileName = os.path.join(app.config['IMAGE_UPLOADS'],image.filename)
                #outFileName = os.path.join(app.config['IMAGE_UPLOADS'],'output.svg')
                predict(inFileName)
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
    p = os.path.join('stitched','stitched.svg')
    return send_file(p,as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)