from PIL import Image
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import os
import cfg
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import random


class CocoDetection(gluon.data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, ann_file):
        self.root = root

        self.coco = COCO(ann_file)

        self.cat_ids = cfg.coco_cat_ids

        self.ids = set()
        for cat_id in self.cat_ids:
            self.ids |= set(self.coco.getImgIds(catIds=cat_id))
        self.ids = list(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img = coco.loadImgs(img_id)[0]
        path = img['file_name']
        w, h = img['width'], img['height']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        target = target_transform(target, h, w, self.coco)

        img, target = img.resize((cfg.size, cfg.size)), target.resize((cfg.size, cfg.size))


        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        #
        # img_small = img.resize((30,30))
        # lbl_small = target.resize((30,30))
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(target)
        # plt.show()
        # lbl_small = np.array(lbl_small)
        # print(lbl_small)

        img = np.array(img).astype(np.float32)
        img = img / 255
        img = mx.image.color_normalize(img, cfg.mean, cfg.std)
        img = np.transpose(img, axes=(2, 0, 1))

        target = np.array(target)

        return img, target

    def __len__(self):
        return len(self.ids)


def target_transform(target, h, w, coco):
    mask = np.zeros((h, w)).astype(np.uint8)
    for t in target:
        cat = t['category_id']
        if cat in cfg.coco_cat_ids:
            m = coco.annToMask(t)
            m *= cfg.coco_map[cat]
            mask[m > 0] = m[m > 0]
    return Image.fromarray(mask)


def coco_test():
    coco_root = os.path.join(cfg.home, 'data', 'COCO', 'val2017')
    anno = os.path.join(cfg.home, 'data', 'COCO',
                        'annotations', 'instances_val2017.json')

    dataset = CocoDetection(coco_root, anno)
    loader = gluon.data.DataLoader(dataset, 4, True)

    for i, data in enumerate(loader):
        print(data)
        break


if __name__ == '__main__':
    coco_test()
