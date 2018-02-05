# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

DATA_DIR, LIST_DIR = "/data2/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012", "../data/pascalvoc12"
TEST_DATA_DIR = "/data2/dataset/pascalvoc2012/VOC2012test/VOCdevkit/VOC2012"

import argparse
import os,sys,cv2
import pprint
from mxnetgo.tensorpack.dataset.pascalvoc12 import PascalVOC12

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
fixed_param_prefix = ['conv0_weight','beta','gamma',]


from symbols.symbol_resnet_deeplabv1_no1024 import resnet101_deeplab_new


def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    # training
    parser.add_argument("--gpu", default="0")
    parser.add_argument('--frequent', help='frequency of logging', default=200, type=int)
    parser.add_argument('--view', action='store_true')
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--load", default="tornadomeet-resnet-101-0000")
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


from mxnetgo.core.module import Predictor
from mxnetgo.myutils.segmentation.segmentation import predict_scaler,visualize_label
from mxnetgo.myutils.stats import MIoUStatistics
from tqdm import tqdm

from mxnetgo.myutils import logger


import os
from tensorpack.dataflow.common import BatchData, MapData
from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes
from mxnetgo.tensorpack.dataset.pascalvoc12 import PascalVOC12
from tensorpack.dataflow.imgaug.misc import  Flip
from tensorpack.dataflow.image import AugmentImageComponents
from tensorpack.dataflow.prefetch import PrefetchDataZMQ
from mxnetgo.myutils.segmentation.segmentation import visualize_label
from seg_utils import RandomCropWithPadding,RandomResize




def get_data(name, data_dir, meta_dir, gpu_nums):
    isTrain = name == 'train'
    ds = PascalVOC12(data_dir, meta_dir, name, shuffle=True)


    if isTrain:
        ds = MapData(ds, RandomResize)

    if isTrain:
        shape_aug = [
                     RandomCropWithPadding(args.crop_size,IGNORE_LABEL),
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
        ds = BatchData(ds, args.batch_size*gpu_nums)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def proceed_test():
    ds = PascalVOC12(TEST_DATA_DIR, LIST_DIR, "test", shuffle=False)
    imagelist = ds.imglist

    def f(ds):
        image = ds
        m = np.array([104, 116, 122])
        const_arr = np.resize(m, (1,1,3))  # NCHW
        image = image - const_arr
        return image

    ds = MapData(ds, f)
    ds = BatchData(ds,1)
    ctx = [mx.gpu(int(i)) for i in args.gpu.split(',')]

    sym_instance = resnet101_deeplab_new()
    # infer shape
    val_provide_data = [[("data", (1, 3, tile_height, tile_width))]]
    val_provide_label = [[("softmax_label", (1, 1, tile_height, tile_width))]]
    data_shape_dict = {'data': (1, 3, tile_height, tile_width)
        , 'softmax_label': (1, 1, tile_height, tile_width)}
    eval_sym = sym_instance.get_symbol(NUM_CLASSES, is_train=False, use_global_stats=True)
    sym_instance.infer_shape(data_shape_dict)

    arg_params, aux_params = load_init_param(args.load, process=True)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)
    data_names = ['data']
    label_names = ['softmax_label']

    # create predictor
    predictor = Predictor(eval_sym, data_names, label_names,
                          context=ctx,
                          provide_data=val_provide_data, provide_label=val_provide_label,
                          arg_params=arg_params, aux_params=aux_params)


    from mxnetgo.myutils.fs import mkdir_p
    vis_dir = "deeplabv1freeze_test_result"
    check_dir = os.path.join(vis_dir,"check")
    import shutil
    shutil.rmtree(vis_dir, ignore_errors=True)
    mkdir_p(check_dir)


    _itr = ds.get_data()
    nbatch = 0
    for i in tqdm(range(len(imagelist))):
        data = next(_itr)
        l = imagelist[i]
        filename = os.path.basename(l).rsplit(".", 1)[0]
        print filename
        output_all = predict_scaler(data, predictor,
                                    scales=[0.9,1.0,1.1], classes=NUM_CLASSES,
                                    tile_size=(tile_height, tile_width),
                                    is_densecrf=False, nbatch=nbatch,
                                    val_provide_data=val_provide_data,
                                    val_provide_label=val_provide_label)
        output_all = np.argmax(output_all, axis=0).astype(np.uint8)
        result = output_all[:,:,None]
        cv2.imwrite(os.path.join(vis_dir,"{}.png".format(filename)),result)
        cv2.imwrite(os.path.join(check_dir,"{}.png".format(filename)),np.concatenate((data[0][0], visualize_label(output_all)), axis=1))
        nbatch += 1



def train_net(args, ctx):
    logger.auto_set_dir()


    sym_instance = resnet101_deeplab_new()
    sym = sym_instance.get_symbol(NUM_CLASSES, is_train=True,use_global_stats=False)
    eval_sym_instance = resnet101_deeplab_new()

    eval_sym = eval_sym_instance.get_symbol(NUM_CLASSES, is_train=False,use_global_stats=True)
    gpu_nums = len(ctx)
    input_batch_size = args.batch_size * gpu_nums

    train_data = get_data("train", DATA_DIR, LIST_DIR, len(ctx))
    test_data = get_data("val", DATA_DIR, LIST_DIR, len(ctx))


    # infer shape
    data_shape_dict = {'data':(args.batch_size, 3, args.crop_size[0],args.crop_size[1])
                       ,'label':(args.batch_size, 1, args.crop_size[0],args.crop_size[1])}

    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)


    # load and initialize params
    epoch_string = args.load.rsplit("-",2)[1]
    begin_epoch = 1
    if not args.scratch:
        begin_epoch = int(epoch_string)
        logger.info('continue training from {}'.format(begin_epoch))
        arg_params, aux_params = load_init_param(args.load, convert=True)
    else:
        logger.info(args.load)
        arg_params, aux_params = load_init_param(args.load, convert=True)
        sym_instance.init_weights(arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    data_names = ['data']
    label_names = ['label']

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,context=ctx, fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    fcn_loss_metric = metric.FCNLogLossMetric(args.frequent)
    eval_metrics = mx.metric.CompositeEvalMetric()

    for child_metric in [fcn_loss_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callbacks = [callback.Speedometer(input_batch_size, frequent=args.frequent)]
    #batch_end_callbacks = [mx.callback.ProgressBar(total=train_data.size/train_data.batch_size)]
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
    mod.fit(train_data=train_data, args = args, eval_sym=eval_sym, eval_sym_instance=eval_sym_instance, eval_data=test_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callbacks,
            batch_end_callback=batch_end_callbacks, kvstore=kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch,epoch_scale=EPOCH_SCALE, validation_on_last=validation_on_last)

def view_data(ctx):
        ds = get_data("train", DATA_DIR, LIST_DIR, ctx)
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
    elif args.test:
        proceed_test()
    else:
        train_net(args, ctx)
