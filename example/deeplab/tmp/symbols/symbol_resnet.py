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

    def residual_unit(self, data, num_filter, stride, dim_match, dilation, name, bottle_neck=True, bn_mom=0.9, workspace=512,
                      memonger=False):
        """Return ResNet Unit symbol for building ResNet
        Parameters
        ----------
        data : str
            Input data
        num_filter : int
            Number of output channels
        bnf : int
            Bottle neck channels factor with regard to num_filter
        stride : tupe
            Stride used in convolution
        dim_match : Boolen
            True means channel number between input and output is the same, otherwise means differ
        name : str
            Base name of the operators
        workspace : int
            Workspace used in convolution operator
        """
        if bottle_neck:
            # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5,use_global_stats=self.use_global_stats, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1),
                                       pad=(0, 0),
                                       no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5,use_global_stats=self.use_global_stats, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                       pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv2')

            if dilation <= 1:
                conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                           pad=(1, 1),
                                           no_bias=True, workspace=workspace, name=name + '_conv2')
            else:
                conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride,
                                           pad=(2, 2), dilate=(dilation, dilation), # here we need padding =(2,2),when dilate=2 and stride=2
                                           no_bias=True, workspace=workspace, name=name + '_conv2')

            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5,use_global_stats=self.use_global_stats, momentum=bn_mom, name=name + '_bn3')
            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
            conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1),
                                       no_bias=True,
                                       workspace=workspace, name=name + '_conv3')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0),
                                              no_bias=True,
                                              workspace=workspace, name=name + '_sc')
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            return conv3 + shortcut
        else:
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, use_global_stats=self.use_global_stats, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, use_global_stats=self.use_global_stats, eps=2e-5, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv2')
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                              no_bias=True,
                                              workspace=workspace, name=name + '_sc')
            if memonger:
                shortcut._set_attr(mirror_stage='True')

            return conv2 + shortcut


    def get_symbol(self, num_class, is_train, units=[3, 4, 23, 3], num_stage=4, filter_list=[64, 256, 512, 1024, 2048],b_lr_mult=2.0,w_lr_mult=1.0, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
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
        self.use_global_stats = True

        if self.is_train:
            logger.info("is_train: {}".format(self.is_train))
            logger.info("use_global_stats: {}".format(self.use_global_stats))

        assert(num_unit == num_stage)
        data = mx.sym.Variable(name='data')
        if self.is_train:
            seg_cls_gt = mx.symbol.Variable(name='label')
        data = mx.sym.BatchNorm(data=data, fix_gamma=True, use_global_stats=self.use_global_stats, eps=2e-5, momentum=bn_mom, name='bn_data')

        ## body for imagenet, note that cifar is another different body
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, use_global_stats=self.use_global_stats,eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        dilation = [1,1,2,2]
        for i in range(num_stage):
            body = self.residual_unit(body, filter_list[i+1], (1 if i==0 or i==3 else 2, 1 if i==0 or i==3 else 2), False,
                                      dilation=dilation[i],name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck,
                                      workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = self.residual_unit(body, filter_list[i+1], (1,1), True, dilation=dilation[i], name='stage%d_unit%d' % (i + 1, j + 2),
                                     bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, use_global_stats=self.use_global_stats, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        #end  of resnet



        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=b_lr_mult)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=w_lr_mult)

        fc6 = mx.symbol.Convolution(
            data=relu1, kernel=(1, 1), pad=(0, 0),dilate=(6,6), num_filter=num_class, name="fc6", bias=fc6_bias, weight=fc6_weight,
            workspace=workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')


        upsamle_scale = 16# upsample 4X
        croped_score = mx.symbol.Deconvolution(
            data=relu_fc6, num_filter=num_class, kernel=(upsamle_scale*2, upsamle_scale*2), stride=(upsamle_scale, upsamle_scale), num_group=num_class, no_bias=True,
            name='upsampling', attr={'lr_mult': '0.0'}, workspace=workspace)

        #magic Cropping
        croped_score = mx.symbol.Crop(*[croped_score, data], offset=(8, 8), name='croped_score')

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
