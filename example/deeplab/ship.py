# Author: Tao Hu <taohu620@gmail.com>

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

import _init_paths


import train
#import test

if __name__ == "__main__":
    train.main()

