import os
import cv2
import json
import numpy as np
import ast

PATH = "/media/bhux/ssd/grss_dse"

train = PATH + "/dfc2021_dse_train/Train"
val = PATH + "/dfc2021_dse_val/Val"
labels_path = PATH + "/labels"

os.makedirs(labels_path,exist_ok=True)
for tile in range(1,61):
    tileFile = train + 'Tile' + str(tile) + '.txt'
    
    with open(tileFile) as f:
        data = json.load(f)
        
    for y in range(16):
      for x in range(16):
          folderPath = PATH + "/labels/Tile"+ str(tile)+"_"+str(y)+"_"+str(x)
          chunkKey = "Tile"+str(tile)+"_"+str(y)+"_"+str(x)
          
          os.makedirs(folderPath,exist_ok=True)
          
          cElecLabel = ast.literal_eval(data[chunkKey]["elecLabel"])
          cSettLabel = ast.literal_eval(data[chunkKey]["settLabel"])
          
          elecLabelImg = np.zeros((150,150))
          settLabelImg = np.zeros((150,150))
          
          elecLabelImg = [[int(cElecLabel[(i//50)*3 + j//50]==True) for j,_ in enumerate(x)] for i,x in enumerate(elecLabelImg)]
          settLabelImg = [[int(cSettLabel[(i//50)*3 + j//50]==True) for j,_ in enumerate(x)] for i,x in enumerate(settLabelImg)]
          
          newImgPath = folderPath + "/elecLabelImg.tif"
          cv2.imwrite(newImgPath, np.array(elecLabelImg))
          
          newImgPath = folderPath + "/settLabelImg.tif"
          cv2.imwrite(newImgPath, np.array(settLabelImg))