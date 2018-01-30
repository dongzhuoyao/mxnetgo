
LIST_DIR = "../data/cityscapes"
import argparse
import os,sys,cv2
import pprint
from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


IGNORE_LABEL = 255

CROP_HEIGHT = 672
CROP_WIDTH = 672
tile_height = 1024
tile_width = 1024
batch_size = 5 #was 7

EPOCH_SCALE = 18
end_epoch = 9
lr_step_list = [(6, 1e-3), (9, 1e-4)]
NUM_CLASSES = Cityscapes.class_num()
validation_on_last = 2

kvstore = "device"
fixed_param_prefix = []
# conv1: fix res1
# bn_conv1: fix res1's bn
# res2: fix res2
# bn2: fix res2's bn
# gamma: fix all bn's gamma
# beta: fix all bn's beta



def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    parser.add_argument("--gpu", default="1")
    parser.add_argument('--frequent', help='frequency of logging', default=800, type=int)
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
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.imgaug.misc import RandomResize,Flip
from tensorpack.dataflow.image import AugmentImageComponents
from tensorpack.dataflow.prefetch import PrefetchDataZMQ
from mxnetgo.myutils.segmentation.segmentation import visualize_label

from seg_utils import RandomCropWithPadding,RandomResize


def get_data(name, meta_dir, gpu_nums):
    isTrain = name == 'train'
    ds = Cityscapes(meta_dir, name, shuffle=True)

    if isTrain:
        ds = MapData(ds, RandomResize)

    if isTrain:#special augmentation
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
        ds = PrefetchDataZMQ(ds, 3)
    else:
        ds = BatchData(ds, 1)
    return ds




def train_net(args, ctx):
    logger.auto_set_dir()

    from symbols.symbol_resnet import resnet101_deeplab_new
    sym_instance = resnet101_deeplab_new()
    sym = sym_instance.get_symbol(NUM_CLASSES, is_train=True)

    #digraph = mx.viz.plot_network(sym, save_format='pdf')
    #digraph.render()

    # setup multi-gpu
    gpu_nums = len(ctx)
    input_batch_size = args.batch_size * gpu_nums

    train_data = get_data("train", LIST_DIR, len(ctx))
    test_data = get_data("val", LIST_DIR, len(ctx))

    eval_sym_instance = resnet101_deeplab_new()


    # infer max shape
    max_scale = [args.crop_size]
    max_data_shape = [('data', (args.batch_size, 3, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]
    max_label_shape = [('label', (args.batch_size, 1, max([v[0] for v in max_scale]), max([v[1] for v in max_scale])))]

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

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,context=ctx, max_data_shapes=[max_data_shape for _ in xrange(gpu_nums)],
                        max_label_shapes=[max_label_shape for _ in xrange(gpu_nums)], fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    fcn_loss_metric = metric.FCNLogLossMetric(args.frequent)
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

    lr_scheduler = StepScheduler(train_data.size()*EPOCH_SCALE,lr_step_list)

    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': 2.5e-4,
                      'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}



    logger.info("epoch scale = {}".format(EPOCH_SCALE))
    mod.fit(train_data=train_data, args = args, eval_sym_instance=eval_sym_instance, eval_data=test_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callbacks,
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