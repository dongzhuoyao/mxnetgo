#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fs.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
from six.moves import urllib
import errno
import tqdm
from . import logger


__all__ = ['mkdir_p', 'download', 'recursive_walk']


def mkdir_p(dirname):
    """ Make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def download(url, dir, filename=None):
    """
    Download URL to a directory.
    Will figure out the filename automatically from URL, if not given.
    """
    mkdir_p(dir)
    if filename is None:
        filename = url.split('/')[-1]
    fpath = os.path.join(dir, filename)

    def hook(t):
        last_b = [0]

        def inner(b, bsize, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    try:
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=hook(t))
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except IOError:
        logger.error("Failed to download {}".format(url))
        raise
    assert size > 0, "Download an empty file!"
    # TODO human-readable size
    print('Succesfully downloaded ' + filename + ". " + str(size) + ' bytes.')
    return fpath


def recursive_walk(rootdir):
    """
    Yields:
        str: All files in rootdir, recursively.
    """
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)




if __name__ == '__main__':
    download('http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz', '.')
