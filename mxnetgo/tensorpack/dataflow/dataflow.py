# Author: Tao Hu <taohu620@gmail.com>
import cv2
from tensorpack.dataflow.common import BatchData, MapData, ProxyDataFlow
from tensorpack.utils.serialize import dumps,loads
from mxnetgo.myutils import logger
import numpy as np
import pprint

def ImageDecode(ds):
    key, obj = ds
    obj = loads(obj)
    def func(im_data,flag):
            img = cv2.imdecode(im_data, flag)
            return img
    return func(obj[0], cv2.IMREAD_COLOR), func(obj[1], cv2.IMREAD_GRAYSCALE)


class FastBatchData(ProxyDataFlow):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The batch can be either a list of original components, or (by default)
    a numpy array of original components.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guranteed to have the same batch size.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(FastBatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = int(batch_size)
        self.remainder = remainder
        self.use_list = use_list

    def size(self):
        ds_size = self.ds.size()
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield FastBatchData._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield FastBatchData._aggregate_batch(holder, self.use_list)

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                if type(dt) in [int, bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    try:
                        tp = dt.dtype
                    except AttributeError:
                        raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                try:
                    shape = data_holder[0][k].shape
                    new_shape = [len(data_holder)]
                    new_shape.extend(list(shape))
                    tmp = np.zeros(tuple(new_shape),dtype=np.int32)
                    for i, x in enumerate(data_holder):
                        tmp[i] = x[k]
                    result.append(tmp)
                    #result.append(np.stack([x[k] for x in data_holder],axis=0))

                    #result.append(
                    #    np.asarray([x[k] for x in data_holder], dtype=tp))
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa
                    logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    if isinstance(dt, np.ndarray):
                        s = pprint.pformat([x[k].shape for x in data_holder])
                        logger.error("Shape of all arrays to be batched: " + s)
                    try:
                        # open an ipython shell if possible
                        import IPython as IP; IP.embed()    # noqa
                    except ImportError:
                        pass
        return result
