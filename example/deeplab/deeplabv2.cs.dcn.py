
LIST_DIR = "data/cityscapes"
import argparse
import os,sys,cv2
import pprint

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


IGNORE_LABEL = 255

CROP_HEIGHT = 1024
CROP_WIDTH = 1024
tile_height = 1024
tile_width = 1024

EPOCH_SCALE = 4
end_epoch = 10
lr_step_list = [(6, 1e-3), (10, 1e-4)]
NUM_CLASSES = 19
kvstore = "device"
fixed_param_prefix = ["conv1", "bn_conv1", "res2", "bn2", "gamma", "beta"]
symbol_str = "resnet_v1_101_deeplab_dcn"


def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    parser.add_argument("--gpu", default="4")
    parser.add_argument('--frequent', help='frequency of logging', default=800, type=int)
    parser.add_argument('--view', action='store_true')
    parser.add_argument("--validation", action="store_true")
    #parser.add_argument("--load", default="train_log/deeplabv2.train.cs/mxnetgo-0080")
    parser.add_argument("--load", default="resnet_v1_101-0000")
    parser.add_argument("--scratch", action="store_true" )
    parser.add_argument('--batch_size', default=1)
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
from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes
from tensorpack.dataflow.imgaug.misc import RandomResize,Flip,RandomCropWithPadding
from tensorpack.dataflow.image import AugmentImageComponents
from tensorpack.dataflow.prefetch import PrefetchDataZMQ
from mxnetgo.myutils.segmentation.segmentation import visualize_label



def get_data(name, meta_dir, gpu_nums):
    isTrain = name == 'train'
    ds = Cityscapes(meta_dir, name, shuffle=True)
    if isTrain:#special augmentation
        shape_aug = [RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                            aspect_ratio_thres=0.15),
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


def test_deeplab(ctx):
    #logger.auto_set_dir()
    test_data = get_data("val", LIST_DIR, len(ctx))
    ctx = [mx.gpu(int(i)) for i in args.gpu.split(',')]

    sym_instance = eval(symbol_str)()
    # infer shape
    val_provide_data = [[("data", (1, 3, tile_height, tile_width))]]
    val_provide_label = [[("softmax_label", (1, 1, tile_height, tile_width))]]
    data_shape_dict = {'data': (1, 3, tile_height, tile_width)
        , 'softmax_label': (1, 1, tile_height, tile_width)}
    eval_sym = sym_instance.get_symbol(NUM_CLASSES, is_train=False)
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

    if args.vis:
        from mxnetgo.myutils.fs import mkdir_p
        vis_dir = os.path.join(logger.get_logger_dir(),"vis")
        mkdir_p(vis_dir)
    stats = MIoUStatistics(NUM_CLASSES)
    test_data.reset_state()
    nbatch = 0
    for data, label in tqdm(test_data.get_data()):
        output_all = predict_scaler(data, predictor,
                                    scales=[1.0], classes=NUM_CLASSES,
                                    tile_size=(tile_height, tile_width),
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


def train_net(args, ctx):
    logger.auto_set_dir()

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', symbol_str + '.py'), logger.get_logger_dir())#copy file to logger dir for debug convenience

    sym_instance = eval(symbol_str)()
    sym = sym_instance.get_symbol(NUM_CLASSES, is_train=True)

    #digraph = mx.viz.plot_network(sym, save_format='pdf')
    #digraph.render()

    # setup multi-gpu
    gpu_nums = len(ctx)
    input_batch_size = args.batch_size * gpu_nums

    train_data = get_data("train", LIST_DIR, len(ctx))
    test_data = get_data("val", LIST_DIR, len(ctx))

    eval_sym_instance = eval(symbol_str)()


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
    begin_epoch = 0
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
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch,epoch_scale=EPOCH_SCALE, validation_on_last=9)

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
