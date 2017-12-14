DATA_DIR, LIST_DIR = "/data_a/dataset/cityscapes", "data/cityscapes"

import _init_paths
import argparse
import os,sys
import pprint
import cv2
from mxnetgo.myutils.config import config, update_config

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

use_cache = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', default="cfg/deeplab_resnet_v1_101_cityscapes_segmentation_base.yaml", type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    parser.add_argument('--vis', help='image visualization', default=True)

    args = parser.parse_args()
    return args

args = parse_args()

import mxnet as mx
from mxnetgo.myutils import logger
from symbols.resnet_v1_101_deeplab import resnet_v1_101_deeplab
from symbols.resnet_v1_101_deeplab_dcn import resnet_v1_101_deeplab_dcn

from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes
from mxnetgo.core.tester import Predictor
from mxnetgo.myutils.seg.segmentation import predict_scaler,visualize_label
from mxnetgo.myutils.load_model import load_param
from mxnetgo.myutils.stats import MIoUStatistics
from tensorpack.dataflow.common import BatchData
from tqdm import tqdm
import numpy as np

IGNORE_LABEL = 255

arg_params, aux_params = load_param("train_log/deeplabv2.train.cs.1th/mxnetgo", 80, process=True)


def get_data(name, data_dir, meta_dir, config):
    ds = Cityscapes(data_dir, meta_dir, name, shuffle=False)
    ds = BatchData(ds, 1)
    return ds


def test_deeplab():
    logger.auto_set_dir()
    test_data = get_data("val", DATA_DIR, LIST_DIR, config)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    logger.info('testing config:{}\n'.format(pprint.pformat(config)))

    # load symbol and testing data
    from symbols import resnet_v1_101_deeplab
    sym_instance = eval(config.symbol + '.' + config.symbol)()

    # infer shape
    val_provide_data = [[("data", (1L, 3L, config.TEST.tile_height, config.TEST.tile_width))]]
    val_provide_label = [[("softmax_label", (1L, 1L, config.TEST.tile_height, config.TEST.tile_width))]]
    data_shape_dict = {'data': (1L, 3L, config.TEST.tile_height, config.TEST.tile_width)
        , 'softmax_label': (1L, 1L, config.TEST.tile_height, config.TEST.tile_width)}
    eval_sym = sym_instance.get_symbol(config, is_train=False)
    sym_instance.infer_shape(data_shape_dict)
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


if __name__ == '__main__':
    print args
    test_deeplab()
