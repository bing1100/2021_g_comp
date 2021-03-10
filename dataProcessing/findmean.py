import os
import cv2
import json
import numpy as np
import ast

PATH = "/media/bhux/ssd/grss_dse"

train = PATH + "/dfc2021_dse_train/Train"
val = PATH + "/dfc2021_dse_val/Val"
labels_path = PATH + "/labels"

statDict = {}

fileDict = {
    'LC08':{},
    'DNB':{},
    'L2A':{},
    'S1A':{}
}

lc08Bands = [
    'B1.tif',
    'B2.tif',
    'B3.tif',
    'B4.tif',
    'B5.tif',
    'B6.tif',
    'B7.tif',
    'B8.tif',
    'B9.tif',
    'B10.tif',
    'B11.tif',
]

for tileNum in range(1,61):
    folder = PATH + "/dfc2021_dse_train/Train/Tile" + str(tileNum)
    for r, d, f in os.walk(folder):
        for file in f:
            k = file.split("_")
            if k[0] in fileDict.keys():
                if k[-1] in fileDict[k[0]].keys():
                    if k[-2] in fileDict[k[0]][k[-1]].keys():
                        fileDict[k[0]][k[-1]][k[-2]].append((os.path.join(r,file), file))
                    else:
                        fileDict[k[0]][k[-1]][k[-2]] = [(os.path.join(r,file), file)]
                else:
                    fileDict[k[0]][k[-1]] = {k[-2]:[(os.path.join(r,file), file)]}
    
for fk in fileDict.keys():
    statDict[fk] = {}
    for bk in fileDict[fk].keys():
        statDict[fk][bk] = {}
        for dk in fileDict[fk][bk].keys():
            statDict[fk][bk][dk] = {
                'mean':0,
                'stddev':0
            } 
            
for fk in fileDict.keys():
    for bk in fileDict[fk].keys():
        for dk in fileDict[fk][bk].keys():
            fileList = fileDict[fk][bk][dk]
            for file in fileList:
                img = cv2.imread(file[0],-1)
                d = cv2.meanStdDev(img)
                statDict[fk][bk][dk]['mean'] += d[0][0][0]/60
                statDict[fk][bk][dk]['stddev'] += (d[1][0][0]**2)/60
    
with open(PATH + '/stats_train.txt', 'w') as outfile:
    json.dump(statDict, outfile)

                
    