# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

DATA_DIR, LIST_DIR = "/home/hutao/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012", "../data/pascalvoc12"


import argparse
import os,sys,cv2
import pprint
from mxnetgo.tensorpack.dataset.pascalvoc12 import PascalVOC12, PascalVOC12Files

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


IGNORE_LABEL = 255

CROP_HEIGHT = 473
CROP_WIDTH = 473
tile_height = 321
tile_width = 321

batch_size = 11
EPOCH_SCALE = 8
end_epoch = 9
lr_step_list = [(6, 1e-3), (9, 1e-4)]
NUM_CLASSES = PascalVOC12.class_num()
validation_on_last = end_epoch

kvstore = "device"
fixed_param_prefix = []


def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    # training
    parser.add_argument("--gpu", default="5")
    parser.add_argument('--frequent', help='frequency of logging', default=1000, type=int)
    parser.add_argument('--view', action='store_true')
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--scratch", action="store_true" )
    parser.add_argument('--batch_size', default=batch_size)
    parser.add_argument('--class_num', default=NUM_CLASSES)
    parser.add_argument('--kvstore', default=kvstore)
    parser.add_argument('--end_epoch', default=end_epoch)
    parser.add_argument('--epoch_scale', default=EPOCH_SCALE)
    parser.add_argument('--tile_height', default=tile_height)
    parser.add_argument('--tile_width', default=tile_width)

    parser.add_argument('--vis', help='image visualization',  action="store_true")
    args = parser.parse_args()

    args.crop_size = (CROP_HEIGHT, CROP_WIDTH)


    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))

import shutil
import mxnet as mx
import numpy as np
from mxnetgo.core import callback, metric
from mxnetgo.core.module import MutableModule
from mxnetgo.myutils.lr_scheduler import WarmupMultiFactorScheduler,StepScheduler
from mxnetgo.myutils.load_model import load_param,load_init_param



from mxnetgo.myutils import logger


import os
from tensorpack.dataflow.common import BatchData, MapData
from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes
from mxnetgo.tensorpack.dataset.pascalvoc12 import PascalVOC12
from tensorpack.dataflow.imgaug.misc import  Flip
from tensorpack.dataflow.image import AugmentImageComponents
from tensorpack.dataflow.prefetch import PrefetchDataZMQ, MultiThreadMapData
from mxnetgo.myutils.segmentation.segmentation import visualize_label
from seg_utils import RandomCropWithPadding,RandomResize
from mxnetgo.tensorpack.dataflow.dataflow import FastBatchData,ImageDecode



def get_data(name, data_dir, meta_dir, gpu_nums):
    isTrain = True if 'train' in name else False

    def imgread(ds):
        img, label = ds
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        return img, label

    if isTrain:
        #ds = LMDBData('/data2/dataset/cityscapes/cityscapes_train.lmdb', shuffle=True)
        #ds = FakeData([[batch_size, CROP_HEIGHT, CROP_HEIGHT, 3], [batch_size, CROP_HEIGHT, CROP_HEIGHT, 1]], 5000, random=False, dtype='uint8')
        ds = PascalVOC12Files(data_dir, meta_dir, name, shuffle=True)
        ds = MultiThreadMapData(ds,4,imgread, buffer_size= 2)
        #ds = PrefetchDataZMQ(MapData(ds, ImageDecode), 1) #imagedecode is heavy
        ds = MapData(ds, RandomResize)
    else:
        ds = PascalVOC12Files(data_dir, meta_dir, name, shuffle=False)
        ds = MultiThreadMapData(ds, 4, imgread, buffer_size= 2)

    if isTrain:
        shape_aug = [
                     RandomCropWithPadding(args.crop_size,IGNORE_LABEL),
                     Flip(horiz=True),
                     ]
        ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

    def reduce_mean_rgb(ds):
        image, label = ds
        m = np.array([104, 116, 122])
        const_arr = np.resize(m, (1,1,3))  # NCHW
        image = image - const_arr
        return image, label

    def MxnetPrepare(ds):
        data, label = ds
        data = np.transpose(data, (0, 3, 1, 2))  # NCHW
        label = label[:, :, :, None]
        label = np.transpose(label, (0, 3, 1, 2))  # NCHW
        dl = [[mx.nd.array(data[args.batch_size * i:args.batch_size * (i + 1)])] for i in
              range(gpu_nums)]  # multi-gpu distribute data, time-consuming!!!
        ll = [[mx.nd.array(label[args.batch_size * i:args.batch_size * (i + 1)])] for i in
              range(gpu_nums)]
        return dl, ll

    #ds = MapData(ds, reduce_mean_rgb)
    ds = MultiThreadMapData(ds, 4, reduce_mean_rgb, buffer_size=2)

    if isTrain:
        ds = FastBatchData(ds, args.batch_size*gpu_nums)
        ds = MapData(ds, MxnetPrepare)
        #ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds



def train_net(args, ctx):
    logger.auto_set_dir()

    from symbols.gluon_deeplabv2 import resnet101_deeplab_new

    input_batch_size = args.batch_size * len(ctx)

    sym_instance = resnet101_deeplab_new()
    sym = sym_instance.get_symbol(NUM_CLASSES, is_train=True)
    eval_sym_instance = resnet101_deeplab_new()
    eval_sym = eval_sym_instance.get_symbol(NUM_CLASSES, is_train=False)

    train_data = get_data("train_aug", DATA_DIR, LIST_DIR, len(ctx))
    eval_data = get_data("val", DATA_DIR, LIST_DIR, len(ctx))

    # infer shape
    data_shape_dict = {'data':(args.batch_size, 3, args.crop_size[0],args.crop_size[1])
                       ,'label':(args.batch_size, 1, args.crop_size[0],args.crop_size[1])}

    sym_instance.infer_shape(data_shape_dict)



    # load and initialize params
    begin_epoch = 1
    mod = MutableModule(sym, data_names=['data'], label_names=['label'],context=ctx, fixed_param_prefix=fixed_param_prefix)

    # metric
    fcn_loss_metric = metric.FCNLogLossMetric(args.frequent,PascalVOC12.class_num())
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(fcn_loss_metric)

    batch_end_callbacks = [callback.Speedometer(input_batch_size, frequent=args.frequent)]
    epoch_end_callbacks = \
        [mx.callback.module_checkpoint(mod, os.path.join(logger.get_logger_dir(),"mxnetgo"), period=1, save_optimizer_states=True),
         ]

    lr_scheduler = StepScheduler(train_data.size()*EPOCH_SCALE,lr_step_list)

    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': 2.5e-4,
                      'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

    logger.info("epoch scale = {}".format(EPOCH_SCALE))
    mod.fit(train_data=train_data, args = args, eval_sym=eval_sym, eval_sym_instance=eval_sym_instance, eval_data=eval_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callbacks,
            batch_end_callback=batch_end_callbacks, kvstore=kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=None, aux_params=None, begin_epoch=begin_epoch, num_epoch=end_epoch,epoch_scale=EPOCH_SCALE, validation_on_last=validation_on_last)

def view_data(ctx):
        ds = get_data("train_aug", DATA_DIR, LIST_DIR, ctx)
        ds.reset_state()
        for ims, labels in ds.get_data():
            for im, label in zip(ims, labels):
                # aa = visualize_label(label)
                # pass
                cv2.imshow("im", im / 255.0)
                cv2.imshow("raw-label", label)
                cv2.imshow("color-label", visualize_label(label))
                cv2.waitKey(0)

if __name__ == '__main__':
    ctx = [mx.gpu(int(i)) for i in args.gpu.split(',')]
    if args.view:
        view_data(ctx)
    elif args.validation:
        test_deeplab(ctx)
    else:
        train_net(args, ctx)
