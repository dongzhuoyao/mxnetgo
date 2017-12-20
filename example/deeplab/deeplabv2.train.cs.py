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
    parser.add_argument('--cfg', help='experiment configure file name', default="cfg/deeplabv2.cs.yaml", type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=1000, type=int)
    parser.add_argument('--view', action='store_true')
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--load", default="train_log/deeplabv2.train.cs/mxnetgo-0080")
    parser.add_argument('--vis', help='image visualization',  action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))

import shutil
import mxnet as mx
import numpy as np
from mxnetgo.core import callback, metric
from mxnetgo.core.module import MutableModule
from mxnetgo.myutils.lr_scheduler import WarmupMultiFactorScheduler,StepScheduler
from mxnet.lr_scheduler import FactorScheduler
from mxnetgo.myutils.load_model import load_param,load_param_by_model


from mxnetgo.core.tester import Predictor
from mxnetgo.myutils.seg.segmentation import predict_scaler,visualize_label
from mxnetgo.myutils.stats import MIoUStatistics
from tqdm import tqdm

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
EPOCH_SCALE = 1
end_epoch = 80
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


def test_deeplab(ctx):
    #logger.auto_set_dir()
    test_data = get_data("val", DATA_DIR, LIST_DIR, config, len(ctx))
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    logger.info('testing config:{}\n'.format(pprint.pformat(config)))
    sym_instance = eval(config.symbol)()
    # infer shape
    val_provide_data = [[("data", (1, 3, config.TEST.tile_height, config.TEST.tile_width))]]
    val_provide_label = [[("softmax_label", (1, 1, config.TEST.tile_height, config.TEST.tile_width))]]
    data_shape_dict = {'data': (1, 3, config.TEST.tile_height, config.TEST.tile_width)
        , 'softmax_label': (1, 1, config.TEST.tile_height, config.TEST.tile_width)}
    eval_sym = sym_instance.get_symbol(config, is_train=False)
    sym_instance.infer_shape(data_shape_dict)

    arg_params, aux_params = load_param_by_model(args.load, process=True)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)
    data_names = ['data']
    label_names = ['softmax_label']

    # create predictor
    predictor = Predictor(eval_sym, data_names, label_names,
                          context=ctx,
                          provide_data=val_provide_data, provide_label=val_provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    if args.vis:
        from mxnetgo.myutils.fs import mkdir_p
        vis_dir = os.path.join(logger.get_logger_dir(),"vis")
        mkdir_p(vis_dir)
    stats = MIoUStatistics(config.dataset.NUM_CLASSES)
    test_data.reset_state()
    nbatch = 0
    for data, label in tqdm(test_data.get_data()):
        output_all = predict_scaler(data, predictor,
                                    scales=[1.0], classes=config.dataset.NUM_CLASSES,
                                    tile_size=(config.TEST.tile_height, config.TEST.tile_width),
                                    is_densecrf=False, nbatch=nbatch,
                                    val_provide_data=val_provide_data,
                                    val_provide_label=val_provide_label)
        output_all = np.argmax(output_all, axis=0)
        label = np.squeeze(label)
        if args.vis:
            cv2.imwrite(os.path.join(vis_dir,"{}.jpg".format(nbatch)),visualize_label(output_all))
        stats.feed(output_all, label)  # very time-consuming
        nbatch += 1
    logger.info("mIoU: {}, meanAcc: {}, acc: {} ".format(stats.mIoU, stats.mean_accuracy, stats.accuracy))


def train_net(args, ctx, pretrained):
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
    epoch_string = args.load.rsplit("-",2)[1]
    begin_epoch = 0
    if len(epoch_string)==4:
        begin_epoch = int(epoch_string)
        logger.info('continue training from {}'.format(begin_epoch))
        arg_params, aux_params = load_param("deeplabv2.train.cs", begin_epoch, convert=True)
    else:
        logger.info(pretrained)
        arg_params, aux_params = load_param(pretrained, begin_epoch, convert=True)
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

    lr_scheduler = StepScheduler(train_data.size()*EPOCH_SCALE/(config.TRAIN.BATCH_IMAGES*len(ctx)),[(3, 1e-4), (5, 1e-5), (7, 8e-6)])

    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': 1e-4,
                      'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}



    mod.fit(train_data=train_data, eval_sym_instance=eval_sym_instance, config=config, eval_data=test_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callbacks,
            batch_end_callback=batch_end_callbacks, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch,epoch_scale=EPOCH_SCALE)

def view_data(ctx):
        ds = get_data("train", DATA_DIR, LIST_DIR, config,ctx)
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
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    if args.view:
        view_data(ctx)

    elif args.validation:
        test_deeplab(ctx)
    else:
        assert config.TRAIN.BATCH_IMAGES%len(ctx)==0, "Batch size must can be divided by ctx_num"
        train_net(args, ctx, config.network.pretrained)
