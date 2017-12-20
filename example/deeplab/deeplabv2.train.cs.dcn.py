# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

DATA_DIR, LIST_DIR = "/data_a/dataset/cityscapes", "data/cityscapes"

import _init_paths

import argparse
import os,sys,cv2
import pprint

from mxnetgo.myutils.config import config, update_config

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default="cfg/deeplabv2.cs.dcn.yaml", type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))

import shutil
import mxnet as mx
import numpy as np
from mxnetgo.core import callback, metric
from mxnetgo.core.module import MutableModule
from mxnetgo.myutils.lr_scheduler import WarmupMultiFactorScheduler
from mxnetgo.myutils.load_model import load_param




from mxnetgo.myutils import logger
from symbols.resnet_v1_101_deeplab import resnet_v1_101_deeplab
from symbols.resnet_v1_101_deeplab_dcn import resnet_v1_101_deeplab_dcn

import os
from tensorpack.dataflow.common import BatchData, MapData
from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes
from tensorpack.dataflow.imgaug.misc import RandomCropWithPadding,RandomResize, Flip
from tensorpack.dataflow.image import AugmentImageComponents
from tensorpack.dataflow.prefetch import PrefetchDataZMQ
from mxnetgo.myutils.seg.segmentation import visualize_label


IGNORE_LABEL = 255

def get_data(name, data_dir, meta_dir, config, gpu_nums):
    isTrain = name == 'train'
    ds = Cityscapes(data_dir, meta_dir, name, shuffle=True)


    if isTrain:#special augmentation
        shape_aug = [RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                            aspect_ratio_thres=0.15),
                     RandomCropWithPadding((config.TRAIN.CROP_HEIGHT, config.TRAIN.CROP_WIDTH),IGNORE_LABEL),
                     Flip(horiz=True),
                     ]
    else:
        shape_aug = []

    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

    def f(ds):
        image, label = ds
        m = np.array([104, 116, 122])
        const_arr = np.resize(m, (1,1,3))  # NCHW
        image = image - const_arr
        return image, label

    ds = MapData(ds, f)
    if isTrain:
        ds = BatchData(ds, config.TRAIN.BATCH_IMAGES*gpu_nums)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    logger.auto_set_dir()

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), logger.get_logger_dir())#copy file to logger dir for debug convenience

    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)

    digraph = mx.viz.plot_network(sym, save_format='pdf')
    digraph.render()

    # setup multi-gpu
    gpu_nums = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * gpu_nums

    # print config
    #pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    train_data = get_data("train", DATA_DIR, LIST_DIR, config, len(ctx))
    test_data = get_data("val", DATA_DIR, LIST_DIR, config, len(ctx))

    eval_sym_instance = eval(config.symbol)()


    # infer max shape
    max_scale = [(config.TRAIN.CROP_HEIGHT, config.TRAIN.CROP_WIDTH)]
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]
    max_label_shape = [('label', (config.TRAIN.BATCH_IMAGES, 1, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]

    # infer shape
    data_shape_dict = {'data':(config.TRAIN.BATCH_IMAGES, 3, config.TRAIN.CROP_HEIGHT, config.TRAIN.CROP_WIDTH)
                       ,'label':(config.TRAIN.BATCH_IMAGES, 1, config.TRAIN.CROP_HEIGHT, config.TRAIN.CROP_WIDTH)}

    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    if config.TRAIN.RESUME:
        logger.info('continue training from {}'.format(begin_epoch))
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        logger.info(pretrained)
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        sym_instance.init_weights(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = ['data']
    label_names = ['label']

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,context=ctx, max_data_shapes=[max_data_shape for _ in xrange(gpu_nums)],
                        max_label_shapes=[max_label_shape for _ in xrange(gpu_nums)], fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    fcn_loss_metric = metric.FCNLogLossMetric(config.default.frequent * gpu_nums)
    eval_metrics = mx.metric.CompositeEvalMetric()

    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in [fcn_loss_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callbacks = [callback.Speedometer(input_batch_size, frequent=args.frequent)]
    #batch_end_callbacks = [mx.callback.ProgressBar(total=train_data.size/train_data.batch_size)]
    epoch_end_callbacks = \
        [mx.callback.module_checkpoint(mod, os.path.join(logger.get_logger_dir(),"mxnetgo"), period=1, save_optimizer_states=True),
         ]

    # decide learning rate
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * train_data.size() / gpu_nums) for epoch in lr_epoch_diff]
    logger.info('lr: {}, lr_epoch_diff: {}, lr_iters: {}'.format(lr,lr_epoch_diff,lr_iters))

    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)

    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}



    mod.fit(train_data=train_data, eval_sym_instance=eval_sym_instance, config=config, eval_data=test_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callbacks,
            batch_end_callback=batch_end_callbacks, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)



def view_data():
        ds = get_data("train", DATA_DIR, LIST_DIR, config)
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
    if args.view == True:
        view_data()

    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    assert config.TRAIN.BATCH_IMAGES%len(ctx)==0, "Batch size must can be divided by ctx_num"
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)
