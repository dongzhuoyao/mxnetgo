import numpy as np
import cv2

img_path= "/data1/dataset/AerialImageDataset/train/gt/tyrol-w7.tif"
#img_path = "/data1/dataset/jpg_aerial/final/gt/yrol-w7_patch2_0.jpg"

#img_path = "/data1/dataset/jpg_aerial/final/gt/yrol-w10_patch3_4.jpg"
img = cv2.imread(img_path,0)
result = np.unique(img)
pass