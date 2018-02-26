# Author: Tao Hu <taohu620@gmail.com>

from mxnet.io import DataIter,DataBatch
from mxnet.io import DataDesc
import threading
import numpy as np
import mxnet as mx
class Tensorpack2Mxnet(DataIter):
    """The base class for an MXNet data iterator.

    All I/O in MXNet is handled by specializations of this class. Data iterators
    in MXNet are similar to standard-iterators in Python. On each call to `next`
    they return a `DataBatch` which represents the next batch of data. When
    there is no more data to return, it raises a `StopIteration` exception.

    Parameters
    ----------
    batch_size : int, optional
        The batch size, namely the number of items in the batch.

    See Also
    --------
    NDArrayIter : Data-iterator for MXNet NDArray or numpy-ndarray objects.
    CSVIter : Data-iterator for csv data.
    LibSVMIter : Data-iterator for libsvm data.
    ImageIter : Data-iterator for images.
    """
    def __init__(self,itr, provide_data, provide_label,batch_size,gpu_nums):
        self.itr = itr
        self.provide_data = provide_data
        self.provide_label = provide_label
        self.batch_size =batch_size
        self.gpu_nums = gpu_nums
        self.lock = threading.Lock() #http://www.redicecn.com/html/Python/20120619/417.html


    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator to the begin of the data."""
        pass

    def next(self):
        """Get next data batch from iterator.

        Returns
        -------
        DataBatch
            The data of next batch.

        Raises
        ------
        StopIteration
            If the end of the data is reached.
        """
        if self.iter_next():
            with self.lock:
                data, label = next(self.itr)
                data = np.transpose(data, (0, 3, 1, 2))  # NCHW
                label = label[:, :, :, None]
                label = np.transpose(label, (0, 3, 1, 2))  # NCHW
                data = [[mx.nd.array(data[self.batch_size * i:self.batch_size * (i + 1)])] for i in
                      range(self.gpu_nums)]  # multi-gpu distribute data, time-consuming!!!
                label = [[mx.nd.array(label[self.batch_size * i:self.batch_size * (i + 1)])] for i in
                      range(self.gpu_nums)]
                return DataBatch(data=data, label=label, \
                    pad=0, index= None, provide_data=self.provide_data,provide_label=self.provide_label)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        return True

    def getdata(self):
        return self.data

    def getlabel(self):
        return self.label

    def getindex(self):
        return None

    def getpad(self):
        return self.pad

class PrefetchingIter(DataIter):
    """Performs pre-fetch for other data iterators.

    This iterator will create another thread to perform ``iter_next`` and then
    store the data in memory. It potentially accelerates the data read, at the
    cost of more memory usage.

    Parameters
    ----------
    iters : DataIter or list of DataIter
        The data iterators to be pre-fetched.
    rename_data : None or list of dict
        The *i*-th element is a renaming map for the *i*-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data.
    rename_label : None or list of dict
        Similar to ``rename_data``.

    Examples
    --------
    >>> iter1 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> iter2 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> piter = mx.io.PrefetchingIter([iter1, iter2],
    ...                               rename_data=[{'data': 'data_1'}, {'data': 'data_2'}])
    >>> print(piter.provide_data)
    [DataDesc[data_1,(25, 10L),<type 'numpy.float32'>,NCHW],
     DataDesc[data_2,(25, 10L),<type 'numpy.float32'>,NCHW]]
    """
    def __init__(self, iters, rename_data=None, rename_label=None):
        super(PrefetchingIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter > 0
        self.iters = iters
        self.rename_data = rename_data
        self.rename_label = rename_label
        self.batch_size = self.provide_data[0][0][1][0] #self.provide_data[0][1][0]
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for i in self.data_taken:
            i.set()
        self.started = True
        self.current_batch = [None for i in range(self.n_iter)]
        self.next_batch = [None for i in range(self.n_iter)]
        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()
        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for i in self.data_taken:
            i.set()
        for thread in self.prefetch_threads:
            thread.join()

    @property
    def provide_data(self):
        if self.rename_data is None:
            return sum([i.provide_data for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_data
            ] for r, i in zip(self.rename_data, self.iters)], [])

    @property
    def provide_label(self):
        if self.rename_label is None:
            return sum([i.provide_label for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_label
            ] for r, i in zip(self.rename_label, self.iters)], [])

    def reset(self):
        for i in self.data_ready:
            i.wait()
        for i in self.iters:
            i.reset()
        for i in self.data_ready:
            i.clear()
        for i in self.data_taken:
            i.set()

    def iter_next(self):
        for i in self.data_ready:
            i.wait()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            for batch in self.next_batch:
                assert batch.pad == self.next_batch[0].pad, \
                    "Number of entry mismatches between iterators"
            self.current_batch = DataBatch(sum([batch.data for batch in self.next_batch], []),
                                           sum([batch.label for batch in self.next_batch], []),
                                           self.next_batch[0].pad,
                                           self.next_batch[0].index,
                                           provide_data=self.provide_data,
                                           provide_label=self.provide_label)
            for i in self.data_ready:
                i.clear()
            for i in self.data_taken:
                i.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad



