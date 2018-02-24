
LIST_DIR = "../data/mapillary"
base_dir = '/data2/dataset/mapillary'
import argparse
import os,sys,cv2
import pprint
from mxnetgo.tensorpack.dataset.Mapillary.Mapillary import Mapillary, MapillaryFiles

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


IGNORE_LABEL = 255

CROP_HEIGHT = 673
CROP_WIDTH = 673
tile_height = 673
tile_width = 673
batch_size = 5 #was 7

EPOCH_SCALE = 4
end_epoch = 9
lr_step_list = [(6, 1e-3), (9, 1e-4)]
NUM_CLASSES = MapillaryFiles.class_num()
validation_on_last = 2

kvstore = "device"
fixed_param_prefix = ['conv0_weight','stage1','beta','gamma',]


def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    parser.add_argument("--gpu", default="1")
    parser.add_argument('--frequent', help='frequency of logging', default=1000, type=int)
    parser.add_argument('--view', action='store_true')
    parser.add_argument("--validation", action="store_true")
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
from tensorpack.dataflow.imgaug.misc import Flip
from tensorpack.dataflow.imgaug.crop import RandomCrop
from tensorpack.dataflow.image import AugmentImageComponents, AugmentImageComponent
from tensorpack.dataflow.prefetch import PrefetchDataZMQ, PrefetchData, MultiThreadMapData
from tensorpack.dataflow.parallel import MultiThreadPrefetchData, MultiProcessPrefetchData
from tensorpack.dataflow.format import LMDBData
from tensorpack.dataflow import FakeData
from mxnetgo.myutils.segmentation.segmentation import visualize_label
from seg_utils import RandomCropWithPadding,RandomResize, ResizeShortestEdge
from tensorpack.utils.serialize import dumps,loads

from mxnetgo.tensorpack.dataflow.dataflow import FastBatchData,ImageDecode

def get_data(name, meta_dir, gpu_nums):
    isTrain = name == 'train'

    def imgread(ds):
        img, label = ds
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        return img, label

    if isTrain:
        #ds = LMDBData('/data2/dataset/Mapillary/Mapillary_train.lmdb', shuffle=True)
        #ds = FakeData([[batch_size, CROP_HEIGHT, CROP_HEIGHT, 3], [batch_size, CROP_HEIGHT, CROP_HEIGHT, 1]], 5000, random=False, dtype='uint8')
        ds = MapillaryFiles(base_dir, meta_dir, name, shuffle=True)
        ds = MultiThreadMapData(ds,8,imgread)
        #ds = PrefetchDataZMQ(MapData(ds, ImageDecode), 1) #imagedecode is heavy
        ds = MapData(ds, ResizeShortestEdge) # ResizeShortestEdge
        ds = MapData(ds, RandomResize)
    else:
        ds = MapillaryFiles(base_dir, meta_dir, name, shuffle=False)
        ds = MultiThreadMapData(ds, 4, imgread)

    if isTrain:#special augmentation
        shape_aug = [
                     RandomCrop(args.crop_size),
                     Flip(horiz=True),
                     ]
        ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

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

    if isTrain:
        ds = FastBatchData(ds, args.batch_size*gpu_nums)
        #ds = PrefetchDataZMQ(ds, 1)
        ds = MapData(ds, MxnetPrepare)
        #ds = PrefetchData(ds,100, 1)
        #ds = MultiProcessPrefetchData(ds, 100, 2)
        #ds = PrefetchDataZMQ(MyBatchData(ds, args.batch_size*gpu_nums), 6)
    else:
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

    fcn_loss_metric = metric.FCNLogLossMetric(args.frequent, Mapillary.class_num())
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
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': 2.5e-4,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}


    logger.info("epoch scale = {}".format(EPOCH_SCALE))
    mod.fit(train_data=train_dataflow, args = args, eval_sym=eval_sym, eval_sym_instance=eval_sym_instance, eval_data=val_dataflow, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callbacks,
            batch_end_callback=batch_end_callbacks, kvstore=kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch,epoch_scale=EPOCH_SCALE, validation_on_last=validation_on_last)

def view_data(ctx):
        ds = get_data("train", LIST_DIR, ctx)
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
