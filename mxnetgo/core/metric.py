
__all__ = ['FCNLogLossMetric','SegMCELossMetric']

import mxnet as mx
import numpy as np
from mxnetgo.myutils import logger

class FCNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, show_interval):
        super(FCNLogLossMetric, self).__init__('FCNLogLoss')
        self.show_interval = show_interval
        logger.info("start training, loss show interval = {}".format(show_interval))
        self.sum_metric = 0
        self.num_inst = 0

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != 255)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)

        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class SegMCELossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(SegMCELossMetric, self).__init__('SegMCELoss')
        self.cls_loss = 0

    def update(self, labels, preds):
        n,c,h,w = preds.shape
        #pred = preds[0]
        #label = labels[0]

        # label (b, p)
        label = labels.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = preds.asnumpy().reshape((preds.shape[0], preds.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((n*h*w, -1 ))

        # filter with keep_inds
        keep_inds = np.where(label != 255)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        self.cls_loss = np.mean(cls_loss)

    def get(self):
        return self.cls_loss

