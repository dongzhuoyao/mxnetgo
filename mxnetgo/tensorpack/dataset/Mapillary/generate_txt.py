# Author: Tao Hu <taohu620@gmail.com>

import os, glob, sys
from shutil import copyfile
import cv2
from tqdm import tqdm
base_path = "/data2/dataset/mapillary"
target_path = '/data2/dataset/mapillary_resize1000'
dst_path = "."

training = os.path.join(base_path, "training/")
testing = os.path.join(base_path, "testing/")
validation = os.path.join(base_path, "validation/")



training_list = []
testing_list = []
validation_list = []

def resize(img, size=1000, interp = cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    scale = size * 1.0 / min(h, w)
    if h < w:
        newh, neww = size, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), size

    img = cv2.resize(img, (neww, newh), interpolation=interp)
    #label = cv2.resize(label, (neww, newh), interpolation=cv2.INTER_NEAREST)
    return img

def write_file(file_list,data_type,f_p):
    print "processing {}".format(data_type)
    for f in tqdm(file_list):
        #print(f)
        basename = os.path.basename(f)[:-4]
        img_base_path = "{}/images/{}.jpg".format(data_type,basename)
        label_base_path = "{}/labels/{}.png".format(data_type, basename)
        cv2.imwrite(os.path.join(target_path,img_base_path),
                    resize(cv2.imread(os.path.join(base_path, img_base_path),cv2.IMREAD_COLOR),interp= cv2.INTER_LINEAR))
        cv2.imwrite(os.path.join(target_path, label_base_path),
                    resize(cv2.imread(os.path.join(base_path, label_base_path),cv2.IMREAD_GRAYSCALE), interp=cv2.INTER_NEAREST))

        write_line = "{} {}\n".format(img_base_path, label_base_path)
        f_p.write(write_line)
        f_p.flush()


def write_file_single(file_list, data_type, f_p):
    print "processing {}".format(data_type)
    for f in tqdm(file_list):
        #print(f)
        basename = os.path.basename(f)[:-4]
        img_base_path = "{}/images/{}.jpg".format(data_type, basename)
        write_line = "{}\n".format(img_base_path)
        cv2.imwrite(os.path.join(target_path, img_base_path),
                    resize(cv2.imread(os.path.join(base_path, img_base_path),cv2.IMREAD_COLOR), interp=cv2.INTER_LINEAR))

        f_p.write(write_line)
        f_p.flush()


# search files
training_list = glob.glob(os.path.join(base_path,"training/images/*.jpg"))
training_list.sort()

testing_list = glob.glob(os.path.join(base_path,"testing/images/*.jpg"))
testing_list.sort()

validation_list = glob.glob(os.path.join(base_path,"validation/images/*.jpg"))
validation_list.sort()


f_train = open(os.path.join(dst_path,"train.txt"),"w")
f_test = open(os.path.join(dst_path,"test.txt"),"w")
f_val = open(os.path.join(dst_path,"val.txt"),"w")


write_file(training_list, "training", f_train)
write_file_single(testing_list, "testing", f_test)
write_file(validation_list, "validation", f_val)
f_train.close()
f_test.close()
f_val.close()


