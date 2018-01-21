import mxnet as mx
import mxnet.gluon as gluon
import os
from PIL import Image
import cfg
import numpy as np
import augment
from matplotlib import pyplot as plt


class VOCDataset(gluon.data.Dataset):
    def __init__(self, root, split, transform=None):
        super(VOCDataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self._img_path = os.path.join(root, 'img', '{}.jpg')
        self._label_path = os.path.join(root, 'SegmentationClass', '{}.png')
        self.ids = list()

        for line in open(os.path.join(root, split + '_seg.txt')):
            self.ids.append(line.strip())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = self._img_path.format(self.ids[idx])
        img = Image.open(img_path).convert('RGB')
        lbl_path = self._label_path.format(self.ids[idx])
        lbl = Image.open(lbl_path)

        # fig = plt.figure()
        # fig.add_subplot(2, 2, 1)
        # plt.imshow(img)
        # fig.add_subplot(2, 2, 2)
        # plt.imshow(lbl)

        if self.transform:
            img, lbl = self.transform(img, lbl)

        # img_ = mx.nd.transpose(img, (1, 2, 0)).asnumpy()
        # img_ = img_ * np.array(cfg.std)
        # img_ = img_ + np.array(cfg.mean)
        # img_ = (img_ * 255).astype(np.uint8)
        # img_ = Image.fromarray(img_)
        # lbl_ = Image.fromarray(lbl.asnumpy()).convert('L')
        # fig.add_subplot(2, 2, 3)
        # plt.imshow(img_)
        # fig.add_subplot(2, 2, 4)
        # plt.imshow(lbl_)
        # plt.show()

        return img, lbl


def voc_test():
    dataset = VOCDataset(cfg.voc_root, 'train', augment.voc_train)
    print(len(dataset))
    dataloader = gluon.data.DataLoader(dataset, 24, True)

    for data in dataloader:
        print(data)
        break


if __name__ == '__main__':
    voc_test()
