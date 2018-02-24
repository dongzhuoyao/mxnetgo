# Author: Tao Hu <taohu620@gmail.com>

import os, glob, sys
from shutil import copyfile
base_path = "/data2/dataset/mapillary"
dst_path = "."

training = os.path.join(base_path, "training/")
testing = os.path.join(base_path, "testing/")
validation = os.path.join(base_path, "validation/")



training_list = []
testing_list = []
validation_list = []


def write_file(file_list,data_type,f_p):
    for f in file_list:
        print(f)
        basename = os.path.basename(f)[:-4]
        write_line = "{}/images/{}.jpg {}/labels/{}.png\n".format(data_type,basename,data_type, basename)
        f_p.write(write_line)
        f_p.flush()

def write_file_single(file_list, data_type, f_p):
        for f in file_list:
            print(f)
            basename = os.path.basename(f)[:-4]
            write_line = "{}/images/{}.jpg\n".format(data_type, basename)
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


