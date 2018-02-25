# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from mxnetgo.myutils import logger
from tensorpack.dataflow.base import RNGDataFlow

__all__ = ['PascalVOC12', 'PascalVOC12Files']


class PascalVOC12(RNGDataFlow):
    def __init__(self, dir, meta_dir, name,
                 shuffle=None):

        assert name in ['train_aug', 'train', 'val','test'], name
        assert os.path.isdir(dir), dir
        self.reset_state()
        self.dir = dir
        self.name = name

        if shuffle is None:
            shuffle = (name == 'train' or name == 'train_aug')
        self.shuffle = shuffle
        self.imglist = []

        if name == 'train':
            f = open(os.path.join(meta_dir,"train.txt"),"r")
        elif name == 'train_aug':
            f = open(os.path.join(meta_dir, "train_aug.txt"), "r")
        elif name =="val":
            f = open(os.path.join(meta_dir, "val.txt"), "r")
        elif name == "test":
            f = open(os.path.join(meta_dir, "test.txt"), "r")
        else:
            raise
        if name=="test":
            for line in f.readlines():
                self.imglist.append(line.strip("\n"))
            f.close()
        else:
            for line in f.readlines():
                self.imglist.append(line.strip("\n").split(" "))
            f.close()

        #self.imglist = self.imglist[:20]

    def size(self):
        return len(self.imglist)

    @staticmethod
    def class_num():
        return 21

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            if self.name != "test":
                fname, flabel = self.imglist[k]
                fname = os.path.join(self.dir, fname)
                flabel = os.path.join(self.dir,flabel)
                fname = cv2.imread(fname, cv2.IMREAD_COLOR)
                flabel = cv2.imread(flabel, cv2.IMREAD_GRAYSCALE)
                yield [fname, flabel]
            else:
                fname = self.imglist[k]
                fname = os.path.join(self.dir, fname)
                fname = cv2.imread(fname, cv2.IMREAD_COLOR)
                yield [fname]


class PascalVOC12Files(RNGDataFlow):
    def __init__(self, dir, meta_dir, name,
                 shuffle=None):

        assert name in ['train_aug', 'train', 'val','test'], name
        assert os.path.isdir(dir), dir
        self.reset_state()
        self.dir = dir
        self.name = name

        if shuffle is None:
            shuffle = (name == 'train' or name == 'train_aug')
        self.shuffle = shuffle
        self.imglist = []

        if name == 'train':
            f = open(os.path.join(meta_dir,"train.txt"),"r")
        elif name == 'train_aug':
            f = open(os.path.join(meta_dir, "train_aug.txt"), "r")
        elif name =="val":
            f = open(os.path.join(meta_dir, "val.txt"), "r")
        elif name == "test":
            f = open(os.path.join(meta_dir, "test.txt"), "r")
        else:
            raise
        if name=="test":
            for line in f.readlines():
                self.imglist.append(line.strip("\n"))
            f.close()
        else:
            for line in f.readlines():
                self.imglist.append(line.strip("\n").split(" "))
            f.close()

        self.imglist = self.imglist[:60]

    def size(self):
        return len(self.imglist)

    @staticmethod
    def class_num():
        return 21

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            if self.name != "test":
                fname, flabel = self.imglist[k]
                fname = os.path.join(self.dir, fname)
                flabel = os.path.join(self.dir,flabel)
                #fname = cv2.imread(fname, cv2.IMREAD_COLOR)
                #flabel = cv2.imread(flabel, cv2.IMREAD_GRAYSCALE)
                yield [fname, flabel]
            else:
                fname = self.imglist[k]
                fname = os.path.join(self.dir, fname)
                #fname = cv2.imread(fname, cv2.IMREAD_COLOR)
                yield [fname]


if __name__ == '__main__':
    pass