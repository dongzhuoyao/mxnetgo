import _init_paths

import cv2
import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config
def parse_args():
    parser = argparse.ArgumentParser(description='Train R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args
args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
import shutil
import mxnet as mx

from function.train_rcnn import train_rcnn
from mxnetgo.myutils import logger


def main():
    logger.auto_set_dir()
    print ('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), logger.get_logger_dir())

    prefix = os.path.join(logger.get_logger_dir(), 'rfcn')
    logging.info('########## TRAIN rfcn WITH IMAGENET INIT AND RPN DETECTION')
    train_rcnn(config, config.dataset.dataset, config.dataset.image_set, config.dataset.root_path, config.dataset.dataset_path,
               args.frequent, config.default.kvstore, config.TRAIN.FLIP, config.TRAIN.SHUFFLE, config.TRAIN.RESUME,
               ctx, config.network.pretrained, config.network.pretrained_epoch, prefix, config.TRAIN.begin_epoch,
               config.TRAIN.end_epoch, train_shared=False, lr=config.TRAIN.lr, lr_step=config.TRAIN.lr_step,
               proposal=config.dataset.proposal)

if __name__ == '__main__':
    main()
