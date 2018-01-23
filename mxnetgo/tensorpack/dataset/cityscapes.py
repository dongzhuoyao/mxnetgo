# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from mxnetgo.myutils import logger
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['Cityscapes']


class Cityscapes(RNGDataFlow):
    def __init__(self, meta_dir, name,
                 shuffle=None):

        assert name in ['train', 'val'], name
        self.reset_state()
        self.name = name
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.imglist = []

        if name == 'train':
            f = open(os.path.join(meta_dir,"train.txt"),"r")
        elif name=="val":
            f = open(os.path.join(meta_dir, "val.txt"), "r")
        elif name=="test":
            f = open(os.path.join(meta_dir, "test.txt"), "r")
        else:
            raise

        for line in f.readlines():
            self.imglist.append(line.strip("\n").split(" "))
        f.close()

        #self.imglist = self.imglist[:50]

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, flabel = self.imglist[k]
            #fname = os.path.join(self.dir, fname)
            #flabel = os.path.join(self.dir,flabel)
            fname = cv2.imread(fname, cv2.IMREAD_COLOR)
            flabel = cv2.imread(flabel, cv2.IMREAD_GRAYSCALE)
            yield [fname, flabel]

    @staticmethod
    def class_num():
        return 19



if __name__ == '__main__':
    pass