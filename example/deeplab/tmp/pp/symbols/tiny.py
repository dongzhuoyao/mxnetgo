'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx
from mxnetgo.myutils.symbol import Symbol
from mxnetgo.myutils import logger


class resnet101_deeplab_new(Symbol):
    def __init__(self):
        pass


    def get_symbol(self, num_class, is_train, use_global_stats, units=[3, 4, 23, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048],b_lr_mult=2.0,w_lr_mult=1.0, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
        """Return ResNet symbol of cifar10 and imagenet
        Parameters
        ----------
        units : list
            Number of units in each stage
        num_stage : int
            Number of stage
        filter_list : list
            Channel size of each stage
        num_class : int
            Ouput size of symbol
        dataset : str
            Dataset type, only cifar10 and imagenet supports
        workspace : int
            Workspace used in convolution operator
        """
        num_unit = len(units)
        self.is_train = is_train
        self.use_global_stats = use_global_stats

        if self.is_train:
            logger.info("is_train: {}".format(self.is_train))
            logger.info("use_global_stats: {}".format(self.use_global_stats))

        assert(num_unit == num_stage)
        data = mx.sym.Variable(name='data')
        if self.is_train:
            seg_cls_gt = mx.symbol.Variable(name='label')
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, use_global_stats=self.use_global_stats, eps=2e-5, momentum=bn_mom, name='bn_data',cudnn_off=True)

        ## body for imagenet, note that cifar is another different body
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, use_global_stats=self.use_global_stats,eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')


        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=b_lr_mult)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=w_lr_mult)

        fc6 = mx.symbol.Convolution(
            data=body, kernel=(1, 1), pad=(0, 0),dilate=(6,6), num_filter=num_class, name="fc6", bias=fc6_bias, weight=fc6_weight,
            workspace=workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')


        upsamle_scale = 2# upsample 4X
        croped_score = mx.symbol.Deconvolution(
            data=relu_fc6, num_filter=num_class, kernel=(upsamle_scale*2, upsamle_scale*2), stride=(upsamle_scale, upsamle_scale), num_group=num_class, no_bias=True,
            name='upsampling', attr={'lr_mult': '0.0'}, workspace=workspace)

        #magic Cropping
        croped_score = mx.symbol.Crop(*[croped_score, data], offset=(3, 3), name='croped_score')

        if is_train:
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                              use_ignore=True, ignore_label=255, name="softmax")
        else:
            softmax = mx.symbol.SoftmaxOutput(data=croped_score, normalization='valid', multi_output=True, use_ignore=True,
                                          ignore_label=255, name="softmax")

        self.sym = softmax
        return softmax

    def init_weights(self, arg_params, aux_params):
        origin_arg_params = arg_params.copy()
        origin_aux_params = aux_params.copy()

        arg_params['fc6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc6_weight'])
        arg_params['fc6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_bias'])
        #arg_params['score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score_weight'])
        #arg_params['score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['score_bias'])
        arg_params['upsampling_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['upsampling_weight'])
        init = mx.init.Initializer()
        init._init_bilinear('upsample_weight', arg_params['upsampling_weight'])

        delta_arg_params = list(set(arg_params.keys()) - set(origin_arg_params.keys()))
        delta_aux_params = list(set(aux_params.keys()) - set(origin_aux_params.keys()))

        logger.info("arg_params initialize manually: {}".format(','.join(sorted(delta_arg_params))))
        logger.info("aux_params initialize manually: {}".format(','.join(sorted(delta_aux_params))))
