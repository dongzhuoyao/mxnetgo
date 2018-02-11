# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Network"
'''

# download url: http://data.dmlc.ml/mxnet/models/imagenet/resnext/101-layers/resnext-101-64x4d-0000.params
# Figure 3.(c)

import mxnet as mx
import numpy as np
from mxnetgo.myutils import logger
from mxnetgo.myutils.symbol import Symbol

class resnext(Symbol):
    def __init__(self):
        pass
    def residual_unit(self, data, num_filter, stride, dim_match, name, use_global_stats, dilation, bottle_neck=True, num_group=32, bn_mom=0.9, workspace=256, memonger=False):
        """Return ResNet Unit symbol for building ResNet
        Parameters
        ----------
        data : str
            Input data
        num_filter : int
            Number of output channels
        bnf : int
            Bottle neck channels factor with regard to num_filter
        stride : tuple
            Stride used in convolution
        dim_match : Boolean
            True means channel number between input and output is the same, otherwise means differ
        name : str
            Base name of the operators
        workspace : int
            Workspace used in convolution operator
        """
        if bottle_neck:
            # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper

            conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                          no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = mx.sym.BatchNorm(data=conv1, use_global_stats=use_global_stats,  fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

            if dilation <= 1:
                conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.5), num_group=num_group,
                                           kernel=(3, 3), stride=stride, pad=(1, 1),
                                           no_bias=True, workspace=workspace, name=name + '_conv2')

            else:
                conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.5), num_group=num_group,
                                           kernel=(3, 3), stride=stride, pad=(dilation,dilation),dilate=(dilation,dilation),
                                           no_bias=True, workspace=workspace, name=name + '_conv2')

            bn2 = mx.sym.BatchNorm(data=conv2,use_global_stats=use_global_stats,  fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')


            conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                       workspace=workspace, name=name + '_conv3')
            bn3 = mx.sym.BatchNorm(data=conv3, use_global_stats=use_global_stats, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

            if dim_match:
                shortcut = data
            else:
                shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                                workspace=workspace, name=name+'_sc')
                shortcut = mx.sym.BatchNorm(data=shortcut_conv,use_global_stats=use_global_stats,  fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

            if memonger:
                shortcut._set_attr(mirror_stage='True')
            eltwise =  bn3 + shortcut
            return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')






        else:

            conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                          no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = mx.sym.BatchNorm(data=conv1,use_global_stats=use_global_stats,  fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')


            conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                          no_bias=True, workspace=workspace, name=name + '_conv2')
            bn2 = mx.sym.BatchNorm(data=conv2, use_global_stats=use_global_stats, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')

            if dim_match:
                shortcut = data
            else:
                shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                                workspace=workspace, name=name+'_sc')
                shortcut = mx.sym.BatchNorm(data=shortcut_conv, use_global_stats=use_global_stats, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')

            if memonger:
                shortcut._set_attr(mirror_stage='True')
            eltwise = bn2 + shortcut
            return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')

    def resnext(self, units, num_stages, filter_list, num_classes, is_train,use_global_stats, num_group, bottle_neck=True,b_lr_mult=2.0,w_lr_mult=1.0, bn_mom=0.9, workspace=256,  memonger=False):


        if is_train:
            logger.info("is_train: {}".format(is_train))
            logger.info("use_global_stats: {}".format(use_global_stats))

        num_unit = len(units)
        assert(num_unit == num_stages)

        if is_train:
            seg_cls_gt = mx.symbol.Variable(name='label')

        data = mx.sym.Variable(name='data')
        data = mx.sym.identity(data=data, name='id')

        data = mx.sym.BatchNorm(data=data, use_global_stats=use_global_stats, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
                          # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body,use_global_stats=use_global_stats,  fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        dilation = [1, 1, 2, 4]
        for i in range(num_stages):
            body = self.residual_unit(body, filter_list[i+1], (1 if i==0 or i==3 else 2, 1 if i==0 or i==3 else 2), False,use_global_stats=use_global_stats,
                                      dilation=1, name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, num_group=num_group,
                                 bn_mom=bn_mom, workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = self.residual_unit(body, filter_list[i+1], (1,1), True,dilation=dilation[i],  name='stage%d_unit%d' % (i + 1, j + 2),use_global_stats=use_global_stats,
                                     bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)

        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=b_lr_mult)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=w_lr_mult)

        fc6 = mx.symbol.Convolution(data=body, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6",
                                    bias=fc6_bias, weight=fc6_weight, workspace=workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=b_lr_mult)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=w_lr_mult)

        score = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1), dilate=(6,6), pad=(0, 0), num_filter=num_classes, name="score",
                                      bias=score_bias, weight=score_weight, workspace=workspace)

        upsamle_scale = 16  # upsample 4X
        croped_score = mx.symbol.Deconvolution(
            data=score, num_filter=num_classes, kernel=(upsamle_scale * 2, upsamle_scale * 2),
            stride=(upsamle_scale, upsamle_scale), num_group=num_classes, no_bias=True,
            name='upsampling', attr={'lr_mult': '0.0'}, workspace=workspace)

        # magic Cropping
        croped_score = mx.symbol.Crop(*[croped_score, data], offset=(8, 8), name='croped_score')

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

    def get_symbol(self, num_classes, is_train, use_global_stats,  num_layers=101, num_group=32, conv_workspace=256, **kwargs):
        """
        Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
        Original author Wei Wu
        """
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

        return self.resnext(units      = units,
                            num_stages  = num_stages,
                            is_train=is_train,
                            use_global_stats = use_global_stats,
                            filter_list = filter_list,
                            num_classes = num_classes,
                            num_group   = num_group,
                            bottle_neck = bottle_neck,
                            workspace   = conv_workspace)

    def init_weights(self, arg_params, aux_params):
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
