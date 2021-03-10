import os
import cv2
import json
import numpy as np
import ast
from pathos.multiprocessing import ProcessingPool as Pool

NUMPROCESS = 8
PATH = "/media/bhux/ssd/grss_dse"
BANDS = ['B04','B03','B02']
DATA = 'L2A'
PANSHARPEN = True

train = PATH + "/dfc2021_dse_train/Train"
val = PATH + "/dfc2021_dse_val/Val"
labels_path = PATH + "/labels"

def checkBands(file):
    for b in BANDS:
        if b in file:
            return True
    return False

def c(pix,mean,stddev):
    d = (pix - mean)/stddev
    if d < 0:
        return round(max(0,(d+1)*63.75))
    return round(min(255,(d+1)*63.75))

with open(PATH + '/stats_train.txt') as f:
    stats = json.load(f)

# def processImg(tile):
for tile in range(1,2):
    for y in range(6,7):
        for x in range(6,7):
            folderPath = PATH + '/chunked/dfc2021_dse_train/Train/Tile'+str(tile)+"_"+str(y)+"_"+str(x)
            files = []
            # r=root, d=directories, f = files
            for r, d, f in os.walk(folderPath):
                for file in f:
                    if DATA in file and checkBands(file) and 'adj' not in file:
                        files.append((os.path.join(r,file), file))
            
            k = files[0][1].split("_")
            keys = stats[k[0]][k[-1]].keys()
            
            for date in stats[k[0]][k[-1]].keys():
                nImg = np.zeros((150,150,3))
                
                axis = 0
                for file in files:
                    print(file, date)
                    if date in file[1]:
                        print("here" + file[0])
                        k = file[1].split("_")
                        mean = stats[k[0]][k[-1]][date]['mean']
                        stddev = stats[k[0]][k[-1]][date]['stddev']**(0.5)
                        
                        img = cv2.imread(file[0], -1)
                        print(file)
                        nImg[:,:,axis] = np.vectorize(lambda px: c(px,mean,stddev))(np.array(img))
                        #nImg[:,:,axis] = np.array(img)
                        axis+=1
                
                newImgPath = folderPath + "/adj_" + DATA + date + ''.join(BANDS) + '.tif'
                cv2.imwrite(newImgPath, nImg.astype(np.float32))
                
# p = Pool(NUMPROCESS)
# results = p.map(processImg, list(range(1,2)))