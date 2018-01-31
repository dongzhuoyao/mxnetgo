# Author: Tao Hu <taohu620@gmail.com>
import os

#prefix = "/data1/dataset/AerialImageCroppedDataset/train"
#prefix_gt = "/data1/dataset/AerialImageCroppedDataset/train_gt"

#f = open("train.txt")
#f_result = open("train_result.txt","w")


prefix = "/data1/dataset/AerialImageCroppedDataset/val"
prefix_gt = "/data1/dataset/AerialImageCroppedDataset/val_gt"

f = open("val.txt")
f_result = open("val_result.txt","w")

for line in f.readlines():
    line = line.strip("\n")
    f_result.write("{} {}\n".format(os.path.join(prefix,line),os.path.join(prefix_gt,line)))

f_result.close()
f.close()
