# Author: Tao Hu <taohu620@gmail.com>
import cv2

seg_gt = cv2.imread("/data_a/dataset/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png",0)
ignore_index = seg_gt != 255
seg_gt = seg_gt[ignore_index]
pass
