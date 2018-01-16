# Author: Tao Hu <taohu620@gmail.com>
from tensorpack.utils import logger
import numpy as np
import cv2

def RandomResize(ds,xrange=(0.7, 1.5), yrange=(0.7, 1.5),aspect_ratio_thres=0.15,minimum=(0,0)):
    image = ds[0]
    label = ds[1]

    assert aspect_ratio_thres >= 0
    if aspect_ratio_thres == 0:
        assert xrange == yrange

    def is_float(tp):
        return isinstance(tp[0], float) or isinstance(tp[1], float)

    assert is_float(xrange) == is_float(yrange), "xrange and yrange has different type!"
    _is_scale = is_float(xrange)

    def _get_augment_params(img):
        cnt = 0
        h, w = img.shape[:2]

        def get_dest_size():
            if _is_scale:
                sx = np.random.uniform(xrange[0],xrange[1],size=[])
                if aspect_ratio_thres == 0:
                    sy = sx
                else:
                    sy = np.random.uniform(yrange[0],yrange[1],size=[])
                destX = max(sx * w, minimum[0])
                destY = max(sy * h, minimum[1])
            else:
                sx = np.random.uniform(xrange[0],xrange[1],size=[])
                if aspect_ratio_thres == 0:
                    sy = sx * 1.0 / w * h
                else:
                    sy = np.random.uniform(yrange[0],yrange[1],size=[])
                destX = max(sx, minimum[0])
                destY = max(sy, minimum[1])
            return (int(destX + 0.5), int(destY + 0.5))

        while True:
            destX, destY = get_dest_size()
            if aspect_ratio_thres > 0:  # don't check when thres == 0
                oldr = w * 1.0 / h
                newr = destX * 1.0 / destY
                diff = abs(newr - oldr) / oldr
                if diff >= aspect_ratio_thres + 1e-5:
                    cnt += 1
                    if cnt > 50:
                        logger.warn("RandomResize failed to augment an image")
                        return h, w, h, w
                        break
                    continue
                return h, w, destY, destX

    h, w, destY, destX = _get_augment_params(image)
    image = cv2.resize(image, (destY, destX),interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (destY,destX),interpolation=cv2.INTER_NEAREST)
    return [image,label]






