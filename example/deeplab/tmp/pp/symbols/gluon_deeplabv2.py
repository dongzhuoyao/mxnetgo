'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx
from mxnetgo.myutils.symbol_gluon import Symbol
from mxnetgo.myutils import logger
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.nn import HybridSequential
from mxnet import gluon


class DilatedBottleneck(HybridBlock):
    def __init__(self, channels, strides, dilation=2, in_channels=0):
        super(DilatedBottleneck, self).__init__()
        self.body = HybridSequential(prefix="dialted-conv")
        self.body.add(nn.Conv2D(channels=channels // 4, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels=channels // 4, kernel_size=3, strides=strides, padding=dilation, dilation=dilation, use_bias=False, in_channels=channels // 4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
    def hybrid_forward(self, F, x):
        residual = x
        x = self.body(x)
        x = F.Activation(residual + x, act_type="relu")
        return x

class Bottleneck(HybridBlock):
    def __init__(self, channels, strides, in_channels=0):
        super(Bottleneck, self).__init__()
        self.body = HybridSequential(prefix="")
        self.body.add(nn.Conv2D(channels=channels // 4, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels=channels // 4, kernel_size=3, strides=strides, padding=1, use_bias=False, in_channels=channels // 4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        self.downsample = nn.HybridSequential()
        self.downsample.add(nn.Conv2D(channels=channels, kernel_size=1, strides=strides, use_bias=False, in_channels=in_channels))
        self.downsample.add(nn.BatchNorm())
    def hybrid_forward(self, F, x):
        residual = self.downsample(x)
        x = self.body(x)
        x = F.Activation(residual + x, act_type="relu")
        return x


class ASPP(HybridBlock):
    def __init__(self):
        super(ASPP, self).__init__()
        gap_kernel = 16
        self.aspp0 = nn.HybridSequential()
        self.aspp0.add(nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0))
        self.aspp0.add(nn.BatchNorm())
        self.aspp1 = self._make_aspp(6)
        self.aspp2 = self._make_aspp(12)
        self.aspp3 = self._make_aspp(18)
        """
        self.gap = nn.HybridSequential()
        self.gap.add(nn.AvgPool2D(pool_size=gap_kernel, strides=1))
        self.gap.add(nn.Conv2D(channels=256, kernel_size=1))
        self.gap.add(nn.BatchNorm())
        upsampling = nn.Conv2DTranspose(channels=256, kernel_size=gap_kernel*2, strides=gap_kernel, padding=gap_kernel/2, weight_initializer=mx.init.Bilinear(), use_bias=False, groups=256)
        upsampling.collect_params().setattr("lr_mult", 0.0)
        self.gap.add(upsampling)
        """
        self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
        self.concurent.add(self.aspp0)
        self.concurent.add(self.aspp1)
        self.concurent.add(self.aspp2)
        self.concurent.add(self.aspp3)
        #self.concurent.add(self.gap)
        self.fire = nn.HybridSequential()
        self.fire.add(nn.Conv2D(channels=256, kernel_size=1))
        self.fire.add(nn.BatchNorm())
    def hybrid_forward(self, F, x):
        return self.fire(self.concurent(x))
    def _make_aspp(self, dilation):
        aspp = nn.HybridSequential()
        aspp.add(nn.Conv2D(channels=256, kernel_size=3, strides=1, dilation=dilation, padding=dilation))
        aspp.add(nn.BatchNorm())
        return aspp


class resnet101_deeplab_new(Symbol):
    def __init__(self):
        pass


    def get_symbol(self,class_num, is_train, pretrained=True):

        self.is_train = is_train
        self.use_global_stats = not is_train

        if self.is_train:
            logger.info("is_train: {}".format(self.is_train))
            logger.info("use_global_stats: {}".format(self.use_global_stats))
        data = mx.sym.Variable(name='data')
        if self.is_train:
            seg_cls_gt = mx.symbol.Variable(name='label')

        resnet = gluon.model_zoo.vision.resnet101_v1(pretrained=pretrained)
        net = nn.HybridSequential()
        for layer in resnet.features[:6]:
            net.add(layer)
        with net.name_scope():
            net.add(Bottleneck(1024, strides=2, in_channels=512))
            for _ in range(23):
                net.add(DilatedBottleneck(channels=1024, strides=1, dilation=2, in_channels=1024))
            net.add(nn.Conv2D(channels=2048, kernel_size=1, strides=1, padding=0))
            for _ in range(3):
                net.add(DilatedBottleneck(channels=2048, strides=1, dilation=4, in_channels=2048))
            net.add(ASPP())
            upsampling = nn.Conv2DTranspose(channels=class_num, kernel_size=32, strides=16, padding=8,
                                            weight_initializer=mx.init.Bilinear(), use_bias=False)
            upsampling.collect_params().setattr("lr_mult", 0.0)
            net.add(upsampling)
            net.add(nn.BatchNorm())

        data = mx.sym.var('data')
        out = net(data)

        croped_score = mx.symbol.Crop(*[out, data], offset=(4, 4), name='croped_score')

        if is_train:
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid',
                                              multi_output=True,
                                              use_ignore=True, ignore_label=255, name="softmax")
        else:
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, normalization='valid', multi_output=True,
                                              use_ignore=True,
                                              ignore_label=255, name="softmax")
        self.sym = softmax
        return softmax


    def init_weights(self, arg_params, aux_params):
        """
        origin_arg_params = arg_params.copy()
        origin_aux_params = aux_params.copy()

        arg_params['fc6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc6_weight'])
        arg_params['fc6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_bias'])
        arg_params['score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score_weight'])
        arg_params['score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['score_bias'])
        arg_params['upsampling_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['upsampling_weight'])
        init = mx.init.Initializer()
        init._init_bilinear('upsample_weight', arg_params['upsampling_weight'])


        delta_arg_params = list(set(arg_params.keys()) - set(origin_arg_params.keys()))
        delta_aux_params = list(set(aux_params.keys()) - set(origin_aux_params.keys()))

        logger.info("arg_params initialize manually: {}".format(','.join(sorted(delta_arg_params))))
        logger.info("aux_params initialize manually: {}".format(','.join(sorted(delta_aux_params))))
         """
