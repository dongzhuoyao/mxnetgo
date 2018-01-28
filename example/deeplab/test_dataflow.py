# Author: Tao Hu <taohu620@gmail.com>

DATA_DIR, LIST_DIR = "/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012", "data/pascalvoc12"


from tensorpack.dataflow.common import BatchData, MapData
from mxnetgo.tensorpack.dataset.cityscapes import Cityscapes
from mxnetgo.tensorpack.dataset.pascalvoc12 import PascalVOC12
from tensorpack.dataflow.imgaug.misc import RandomResize, Flip
from tensorpack.dataflow.image import AugmentImageComponents
from tensorpack.dataflow.prefetch import PrefetchDataZMQ
from mxnetgo.myutils.segmentation.segmentation import visualize_label
from seg_utils import RandomCropWithPadding

from tqdm import  tqdm
import numpy as np

batch_size = 14
crop_size = (473,473)

def get_data(name, data_dir, meta_dir, gpu_nums):
    isTrain = name == 'train'
    ds = PascalVOC12(data_dir, meta_dir, name, shuffle=True)


    if isTrain:#special augmentation
        shape_aug = [RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                            aspect_ratio_thres=0.15),
                     RandomCropWithPadding(crop_size,255),
                     Flip(horiz=True),
                     ]
    else:
        shape_aug = []

    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

    def f(ds):
        image, label = ds
        m = np.array([104, 116, 122])
        const_arr = np.resize(m, (1,1,3))  # NCHW
        image = image - const_arr
        return image, label

    ds = MapData(ds, f)
    if isTrain:
        ds = BatchData(ds, batch_size*gpu_nums)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds

train_data = get_data("train", DATA_DIR, LIST_DIR, 1)
train_data.reset_state()
_itr = train_data.get_data()

for i in tqdm(range(100)):
    data, label = next(_itr)
    print("next")