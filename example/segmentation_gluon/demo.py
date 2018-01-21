import augment
import numpy as np
import mxnet as mx
from PIL import Image
from  matplotlib import pyplot as plt
from model.lkm import LKM
import os
from data.voc_dataset import VOCDataset
import cfg
import mxnet.autograd as ag
resizer = augment.UnitResize(32, 480)


def demo(net, dataset, m):
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    for i in range(m):
        i = idx[i]
        img, lbl = dataset[i]

        img_, lbl_ = resizer(img, lbl)

        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(img_)
        fig.add_subplot(1, 3, 2)
        plt.imshow(lbl_)

        img_, lbl_ = augment.voc_val(img, lbl)
        img_ = mx.nd.expand_dims(img_, 0)
        with ag.predict_mode():
            pred = net(img_)
        pred = mx.nd.argmax(pred, 1).asnumpy().squeeze()
        pred = pred.astype(np.uint8)

        pred = Image.fromarray(pred)
        fig.add_subplot(1, 3, 3)
        plt.imshow(pred)
        plt.show()


def main():
    lkm = LKM(pretrained=False)
    lkm.load_params(os.path.join('save', 'LKM', 'weights'), ctx=mx.cpu())
    lkm.hybridize()
    dataset = VOCDataset(cfg.voc_root, 'val')
    demo(lkm, dataset, 10)


if __name__ == '__main__':
    main()
