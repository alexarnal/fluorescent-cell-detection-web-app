from concurrent.futures import process
import os
import re
import numpy as np
import pdb


def getCoordsFromSVG(fileName):
    file = open(fileName,'r')
    fileContent = file.readlines()
    file.close()

    #fileContent = fileContent.split('>')
    X=[]
    Y=[]
    viewBox = 0
    height, width = 0,0
    for i,line in enumerate(fileContent): 
        #print(line)
        coord = 0
        if 'rect' in line:
            height = int(re.search('height="(.*)" style', line).group(1))
            width = int(re.search('width="(.*)" height', line).group(1))
        if 'viewBox=' in line:
            viewBox = re.search('viewBox="(.*)"', line)
        if 'circle' in line:
            x = re.search('cx="(.*)" cy', line)
            y = re.search('cy="(.*)" r', line)
            coord = [x.group(1),y.group(1)]
            #pdb.set_trace()
        #elif 'path=' in line:
        #    coord = re.search('"M(.*?)a', line, re.I)
        #    coord = coord.group(1).split(',')   
        else: continue
        X.append(float(coord[0]))
        Y.append(float(coord[1]))
    coords = np.vstack((np.array(X), np.array(Y))).T
    return coords, np.array(viewBox[1].split(' '), dtype='float'), height, width

def startSVG(fileName,dims):
    f = open(fileName+".svg", "w")
    f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %s %s">\n'%(dims[0], dims[1]))
    f.write('<rect width="%s" height="%s" style="fill:none; stroke:#000000; stroke-width:1px"/>\n'%(dims[0], dims[1]))
    f.close()

def saveCoordsSVG(x,y,fileName): 
    f = open(fileName+".svg", "a")
    #for i in range(x.shape[0]):
    f.write('<circle cx="%s" cy="%s" r="%s" style="fill:#000000"/>\n'%(x, y, 3))
    f.close()

def endSVG(fileName):
    f = open(fileName+".svg", "a")
    f.write('</svg>')
    f.close()   

from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


def run(filenames):
    allowable_distance = 9.41 * (99/51) / 2 #2.76 * (99/51) 
    path = './uploads/'
    print("You've successfuly loaded the colocalization script")
    print(f'These are the loaded filenames:\n{filenames}')
    svg_files = os.listdir(path)
    coords, view_boxes, heights, widths, lengths = [], [], [], [], [0]
    for f in svg_files:
        c, vb, h, w = getCoordsFromSVG(path+f)
        coords.append(c)
        view_boxes.append(vb)
        heights.append(h)
        widths.append(w)
        lengths.append(len(c)+lengths[-1])
    lengths = np.array(lengths)
    msg = 'Cannot run colocalization: either height or width of images do not match.'
    assert len(np.unique(widths)) == 1 and len(np.unique(heights)) == 1, msg
    width = widths[0]
    height = heights[0]
    #pdb.set_trace()
    for subset in all_subsets(np.arange(len(coords))): 
        if len(subset) > 1: 
            fname = path
            all_cells = []
            for s in subset:
                fname += filenames[s][:-4]+'-'
                all_cells.extend(coords[s])
            all_cells = np.array(all_cells)
            processed = np.zeros(len(all_cells), dtype = bool)
            mutual = [] #
            for i in range(all_cells.shape[0]):
                start = lengths[lengths>i][0]
                end = lengths[-1]
                if processed[i] == True : continue
                inds = np.array(np.where(processed[start:end]==False))[0]
                difs = np.linalg.norm(all_cells[i] - all_cells[start:end][inds], axis=-1)
                allowable_dif_inds = np.array(np.where(difs<allowable_distance))[0]
                if len(allowable_dif_inds) == len(subset)-1:
                    processed[start+inds[allowable_dif_inds]] = True
                    processed[i] = True
                    cells = np.vstack((all_cells[start+inds][allowable_dif_inds], all_cells[i]))
                    position = np.mean(cells, axis=0)
                    print(len(position), i, allowable_dif_inds, len(inds), np.sum(processed))
                    mutual.append(position)#
                    #pdb.set_trace()
            fname = fname[:-1]
            #pdb.set_trace()
            startSVG(fname,[width,height])
            for m in mutual:
                saveCoordsSVG(m[0],m[1], fname)
            endSVG(fname)
    '''for i in range(len(coords)-1):
        fname = f"{filenames[i][:-4]}-{filenames[i+1][:-4]}.svg"
        startSVG(fname,[widths[i],heights[i]])
        c_0 = coords[i]
        c_1 = coords[i+1]
        mutual = []
        for c0 in c_0:
            for c1 in c_1:
                dif = np.linalg.norm(c0 - c1)
                if dif < half_cell_size:
                    saveCoordsSVG(c0[0],c0[1], fname)
                    mutual.append(c0)
            break
        endSVG(fname)'''

    #pdb.set_trace()
