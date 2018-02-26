base_dir = '/home/hutao/dataset/cityscapes'
LIST_DIR = "../data/cityscapes"
import argparse
import os,sys,cv2
import pprint
from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes, CityscapesFiles

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


IGNORE_LABEL = 255

CROP_HEIGHT = 673
CROP_WIDTH = 673
tile_height = 673
tile_width = 673
batch_size = 5 #was 7

EPOCH_SCALE = 15
end_epoch = 9
init_lr = 1e-3
lr_step_list = [(6, 1e-3), (9, 1e-4)]
NUM_CLASSES = CityscapesFiles.class_num()
validation_on_last = 2

kvstore = "device"
fixed_param_prefix = ['conv0_weight','stage1']


def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    parser.add_argument("--gpu", default="1")
    parser.add_argument('--frequent', help='frequency of logging', default=500, type=int)
    parser.add_argument('--view', action='store_true')
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--test_speed", action="store_true")
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
from mxnetgo.myutils.lr_scheduler import StepScheduler
from mxnetgo.myutils.load_model import load_param,load_init_param


from mxnetgo.core.tester import Predictor
from mxnetgo.myutils.segmentation.segmentation import predict_scaler,visualize_label
from mxnetgo.myutils.stats import MIoUStatistics
from tqdm import tqdm

from mxnetgo.myutils import logger
from symbols.resnet_v1_101_deeplab import resnet_v1_101_deeplab
from symbols.resnet_v1_101_deeplab_dcn import resnet_v1_101_deeplab_dcn

import os
from tensorpack.dataflow.common import BatchData, MapData, ProxyDataFlow
from tensorpack.dataflow.imgaug.misc import RandomResize,Flip
from tensorpack.dataflow.imgaug.crop import RandomCrop
from tensorpack.dataflow.image import AugmentImageComponents
from tensorpack.dataflow.prefetch import PrefetchDataZMQ, PrefetchData, MultiThreadMapData
from tensorpack.dataflow.parallel import MultiThreadPrefetchData, MultiProcessPrefetchData
from tensorpack.dataflow.format import LMDBData
from tensorpack.dataflow import FakeData, TestDataSpeed, imgaug
from mxnetgo.myutils.segmentation.segmentation import visualize_label
from seg_utils import RandomCropWithPadding,RandomResize
from tensorpack.utils.serialize import dumps,loads
from mxnetgo.core.io import Tensorpack2Mxnet, PrefetchingIter

from mxnetgo.tensorpack.dataflow.dataflow import FastBatchData,ImageDecode
import multiprocessing

m = np.array([104, 116, 122])
const_arr = np.resize(m, (1,1,3))  # NCHW

def imgread(ds):
    img, label = ds
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
    return img, label


def get_data(name, meta_dir, gpu_nums):
    isTrain = True if 'train' in name else False

    m = np.array([104, 116, 122])
    const_arr = np.resize(m, (1, 1, 3))  # NCHW
    const_arr = np.zeros((args.crop_size[0],args.crop_size[1],3)) + const_arr #broadcast


    if isTrain:
        #ds = FakeData([[1024, 2048, 3], [ 1024, 2048]], 5000, random=False, dtype='uint8')
        #ds = FakeData([[CROP_HEIGHT, CROP_HEIGHT, 3], [CROP_HEIGHT, CROP_HEIGHT]], 5000,random=False, dtype='uint8')
        ds = CityscapesFiles(base_dir, meta_dir, name, shuffle=True)
        parallel = min(3, multiprocessing.cpu_count())
        augmentors = [
            RandomCropWithPadding(args.crop_size),
            Flip(horiz=True),
        ]
        aug = imgaug.AugmentorList(augmentors)

        def mapf(ds):
            img, label = ds
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            img, params = aug.augment_return_params(img)
            label = aug._augment(label, params)
            img = img - const_arr  # very time-consuming
            return img, label

        #ds = MapData(ds, mapf)
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=500, strict=True)
        #ds = MapData(ds, reduce_mean_rgb)

        ds = BatchData(ds, args.batch_size * gpu_nums)
        #ds = PrefetchDataZMQ(ds, 1)
    else:
        def imgread(ds):
            img, label = ds
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            return [img, label]

        ds = CityscapesFiles(base_dir, meta_dir, name, shuffle=False)
        ds = MapData(ds, imgread)
        ds = BatchData(ds, 1)

    return ds




def train_net(args, ctx):
    logger.auto_set_dir()

    from symbols.symbol_resnet_deeplabv2 import resnet101_deeplab_new
    sym_instance = resnet101_deeplab_new()
    sym = sym_instance.get_symbol(NUM_CLASSES, is_train=True,use_global_stats=False)

    # setup multi-gpu
    gpu_nums = len(ctx)
    input_batch_size = args.batch_size * gpu_nums

    train_dataflow = get_data("train", LIST_DIR, len(ctx))
    val_dataflow = get_data("val", LIST_DIR, len(ctx))

    eval_sym_instance = resnet101_deeplab_new()
    eval_sym = eval_sym_instance.get_symbol(args.class_num, is_train=False,use_global_stats=True)

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
    mod = MutableModule(sym, data_names=['data'], label_names=['label'], context=ctx, fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    fcn_loss_metric = metric.FCNLogLossMetric(args.frequent, Cityscapes.class_num())
    eval_metrics = mx.metric.CompositeEvalMetric()

    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in [fcn_loss_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callbacks = [callback.Speedometer(input_batch_size, frequent=args.frequent)]
    epoch_end_callbacks = \
        [mx.callback.module_checkpoint(mod, os.path.join(logger.get_logger_dir(),"mxnetgo"), period=1, save_optimizer_states=True),
         ]

    lr_scheduler = StepScheduler(train_dataflow.size()*EPOCH_SCALE,lr_step_list)

    # optimizer
    optimizer_params = {
        'learning_rate': init_lr,
        'lr_scheduler': lr_scheduler,
    }


    logger.info("epoch scale = {}".format(EPOCH_SCALE))
    mod.fit(train_data=train_dataflow, args = args, eval_sym=eval_sym, eval_sym_instance=eval_sym_instance, eval_data=val_dataflow, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callbacks,
            batch_end_callback=batch_end_callbacks, kvstore=kvstore,
            optimizer='adam', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch,epoch_scale=EPOCH_SCALE, validation_on_last=validation_on_last)

def view_data(ctx):
    m = np.array([104, 116, 122])
    const_arr = np.resize(m, (1, 1, 3))  # NCHW
    const_arr = np.zeros((args.crop_size[0], args.crop_size[1], 3)) + const_arr  # broadcast

    ds = get_data("train", LIST_DIR, ctx)
    ds.reset_state()
    for ims, labels in ds.get_data():
        for im, label in zip(ims, labels):
            # aa = visualize_label(label)
            # pass
            im += const_arr
            cv2.imshow("im", cv2.resize(im,(500,500)) / 255.0)
            #cv2.imshow("raw-label", label)
            cv2.imshow("color-label", cv2.resize(visualize_label(label),(500,500)))
            cv2.waitKey(20000)


def test_speed():
    train_dataflow = get_data("train", LIST_DIR, len(ctx))
    TestDataSpeed(train_dataflow, size=100).start()

if __name__ == '__main__':
    ctx = [mx.gpu(int(i)) for i in args.gpu.split(',')]
    if args.view:
        view_data(ctx)
    elif args.validation:
        test_deeplab(ctx)
    elif args.test_speed:
        test_speed()
    else:
        train_net(args, ctx)
