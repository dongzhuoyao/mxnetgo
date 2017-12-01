# Author: Tao Hu <taohu620@gmail.com>

import mxnet.gluon.models as models
resnet18 = models.resnet18_v1(pretrained=True)
alexnet = models.alexnet(pretrained=True)