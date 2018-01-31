#-*- coding: UTF-8 -*-
import os
import cv2,os
from tqdm import tqdm
import numpy as np

from shutil import copyfile


# cropRGBImage
def cropImage(filepath,pathDir, outputpath, split_num=5):
    print "start crop images."
    for filename in tqdm(pathDir):
        print filename
        child = os.path.join(filepath, filename)
        im = cv2.imread(child)
        lx,ly,lz = im.shape
        for i in range(0,split_num):
            for j in range(0,split_num):
                crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num, :]
                a = os.path.basename(filename).strip(".tif")
                cv2.imwrite(os.path.join(outputpath,"{}_patch{}_{}.tif".format(a,i,j)), crop_im)

# cropBWImage
def cropBW(filepath,pathDir, outputpath, split_num=5):
    print "start crop gt."
    for filename in tqdm(pathDir):
        #if "yrol-w7" in filename:
            print filename
            child = os.path.join(filepath, filename)
            im = cv2.imread(child,0)
            lx,ly = im.shape
            for i in range(0,split_num):
                for j in range(0,split_num):
                    crop_im = im[i*lx/split_num:(i+1)*lx/split_num, j*ly/split_num:(j+1)*ly/split_num].astype(np.uint8)
                    crop_im = crop_im/255
                    a = os.path.basename(filename).strip(".tif")
                    cv2.imwrite(os.path.join(outputpath, "{}_patch{}_{}.tif".format(a, i, j)), crop_im)



def preprocess(train_data_ratio=0.9):
    src_train_imagepath = "/data1/dataset/AerialImageDataset/train/images"
    src_train_gtpath = "/data1/dataset/AerialImageDataset/train/gt"

    src_test_imagepath = "/data1/dataset/AerialImageDataset/test/images"


    dst_val_image = "/data1/dataset/AerialImage/val/images"
    dst_val_gt = "/data1/dataset/AerialImage/val/gt"
    dst_train_image = "/data1/dataset/AerialImage/train/images"
    dst_train_gt = "/data1/dataset/AerialImage/train/gt"
    dst_test_img = "/data1/dataset/AerialImage/test/images"

    import shutil
    shutil.rmtree(dst_val_image)
    shutil.rmtree(dst_val_gt)
    shutil.rmtree(dst_train_image)
    shutil.rmtree(dst_train_gt)
    shutil.rmtree(dst_test_img)

    os.makedirs(dst_val_image)
    os.makedirs(dst_val_gt)
    os.makedirs(dst_train_image)
    os.makedirs(dst_train_gt)
    os.makedirs(dst_test_img)

    pathDir = os.listdir(src_train_imagepath)
    middle = int(train_data_ratio*len(pathDir))
    train_pathDir = pathDir[:middle]
    val_pathDir = pathDir[middle:]

    print "start test"
    f = open("test.txt", "w+")
    for filename in tqdm(os.listdir(src_test_imagepath)):
        copyfile(os.path.join(src_test_imagepath, filename),os.path.join(dst_test_img, filename))
        child = os.path.join('{}\n'.format(os.path.join(dst_test_img, filename)))
        f.write(child)
    f.close()

    #deal with  validation data
    print "start val"
    f = open("val.txt", "w+")
    for filename in tqdm(val_pathDir):
        # copy image
        copyfile(os.path.join(src_train_imagepath, filename), os.path.join(dst_val_image, filename))
        # copy gt
        img_gt = cv2.imread(os.path.join(src_train_gtpath, filename),0)
        img_gt = img_gt/255
        cv2.imwrite(os.path.join(dst_val_gt, filename),img_gt)
        child = os.path.join('{} {}\n'.format(os.path.join(dst_val_image, filename),
                             os.path.join(dst_val_gt, filename)))
        f.write(child)
    f.close()

    print "start train"
    #deal with train data
    cropImage(src_train_imagepath,train_pathDir,dst_train_image)
    cropBW(src_train_gtpath, train_pathDir, dst_train_gt)
    f = open("train.txt","w")
    for filename in tqdm(os.listdir(dst_train_image)):
        f.write("{} {}\n".format(os.path.join(dst_train_image,filename),os.path.join(dst_train_gt,filename)))
    f.close()

if __name__ == '__main__':
    preprocess(0.9)