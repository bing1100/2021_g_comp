import cv2
import os
from pathos.multiprocessing import ProcessingPool as Pool
from PIL import Image
from numpy import asarray
import json

# Note: Copy this file to the train directory before running. s
NUMPROCESS = 8
NEWWIDTH = 150
NEWHEIGHT = 150
STRIDE = 50
CREATE = True
PATH = "/media/bhux/ssd/grss_dse"

train = PATH + "/dfc2021_dse_train/Train"
val = PATH + "/dfc2021_dse_val/Val"

def findSurround(y, x):
    return [((y),(x)), 
            ((y),(x+1)),
            ((y),(x+2)),
            ((y+1),(x)),
            ((y+1),(x+1)),
            ((y+1),(x+2)),
            ((y+2),(x)),
            ((y+2),(x+1)),
            ((y+2),(x+2))]
    
def bound(tup):
    return (-1<tup[0] and tup[0]<16) and (-1<tup[1] and tup[1]<16)

# Training data
for tile in range(1,61):
    tilePath = train + "/Tile" +str(tile)
    
    data = {}
    labels = asarray(Image.open(tilePath + "/groundTruth.tif"))
    
    for y in range(16):
        for x in range(16):
            pts = findSurround(y, x)
    
            chunkStartPt = (pts[0][0] * 50, pts[0][1] * 50)
            chunkLabels = list(map(lambda a : labels[a] if bound(a) else -1, pts))
            chunkLabelsElec = list(map(lambda a : a < 3 if 0 < a else -1, chunkLabels))
            chunkLabelsSett = list(map(lambda a : (a % 2) > 0 if 0 < a else -1, chunkLabels))
            
            data["Tile"+str(tile)+"_"+str(y)+"_"+str(x)] = {
                "startPt":str(chunkStartPt),
                "elecLabel":str(chunkLabelsElec),
                "settLabel":str(chunkLabelsSett),
            }
    
    with open(train + 'Tile' + str(tile) + '.txt', 'w') as outfile:
        json.dump(data, outfile)

# Validation data
for tile in range(1,20):
    tilePath = val + "/Tile" +str(tile)
    
    data = {}
    
    for y in range(16):
        for x in range(16):
            pts = findSurround(y, x)
    
            chunkStartPt = (pts[0][0] * 50, pts[0][1] * 50)
            
            data["Tile"+str(tile)+"_"+str(y)+"_"+str(x)] = {
                "startPt":str(chunkStartPt),
            }
    
    with open(val + 'Tile' + str(tile) + '.txt', 'w') as outfile:
        json.dump(data, outfile)
        
def processImg(tileNum,type):
    folder = PATH + type + "/" + "Tile" + str(tileNum)
    path = PATH + "/chunked" + type
    
    os.makedirs(path,exist_ok=True)
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder):
        for file in f:
            if '.tif' in file and 'groundTruth' not in file:
                files.append((os.path.join(r,file), file))

    for y in range(16):
        for x in range(16):
            os.makedirs(path + "/Tile"+str(tileNum)+"_"+str(y)+"_"+str(x),exist_ok=True)
    
    for file in files:

        img = cv2.imread(file[0], -1)
        
        img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
        for y in range(16):
            for x in range(16):
                pts = findSurround(y, x)
                i = (pts[0][0] * 50, pts[0][1] * 50)
                
                newImgPath = path + "/Tile"+str(tileNum)+"_"+str(y)+"_"+str(x) +"/" + file[1]
                
                split = img[i[0]:i[0]+NEWHEIGHT, i[1]:i[1]+NEWWIDTH]
                cv2.imwrite(newImgPath, split)
        
        
                       
if CREATE:
    p = Pool(NUMPROCESS)
    results = p.map(lambda a : processImg(a, "/dfc2021_dse_train/Train"), list(range(1,61)))
    results = p.map(lambda a : processImg(a, "/dfc2021_dse_val/Val"), list(range(1,20)))