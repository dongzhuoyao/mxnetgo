#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: segmentation.py
# Author: Tao Hu <taohu620@gmail.com>

import numpy as np
from math import ceil
import cv2,colorsys
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
import mxnet as mx
from ...myutils import logger
import os, sys

sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , 'mxnetgo/myutils/seg' ) ) )

__all__ = ['update_confusion_matrix', 'predict_slider']

# Colour map.
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

# C Support
# Enable the cython support for faster evaluation
# Only tested for Ubuntu 64bit OS
CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        import addToConfusionMatrix
    except:
        CSUPPORT = False


def update_confusion_matrix(pred, label, conf_m, nb_classes, ignore = 255):
    if (CSUPPORT):
        # using cython
        conf_m = addToConfusionMatrix.cEvaluatePair(pred.astype(np.uint8), label.astype(np.uint8), conf_m, nb_classes)
        return conf_m
    else:
        logger.warn("confusion matrix c extension not found, this calculation will be very slow")
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)

        for p, l in zip(flat_pred, flat_label):
            if l == ignore:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                raise
        return conf_m

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = max(target_size[0] - img.shape[0], 0)
    cols_missing = max(target_size[1] - img.shape[1], 0)
    try:
        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    except  Exception,e:
        print str(e)
        pass
    return padded_img, [0,target_size[0]-rows_missing,0,target_size[1] - cols_missing]


def pad_edge(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = max(target_size[0] - img.shape[0], 0)
    cols_missing = max(target_size[1] - img.shape[1], 0)
    padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    return padded_img, [0,target_size[0]-rows_missing,0,target_size[1] - cols_missing]


def visualize_label(label):
    """Color classes a good distance away from each other."""
    h, w = label.shape
    img_color = np.zeros((h, w, 3)).astype('uint8')
    for i in range(0,21):
        img_color[label == i] = label_colours[i]
    return img_color


def predict_slider(full_image, predictor, classes, tile_size,nbatch,val_provide_data,val_provide_label):
    overlap = 1.0/3
    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((full_image.shape[0] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((full_image.shape[1] - tile_size[1]) / stride) + 1)
    full_probs = np.zeros((classes,full_image.shape[0], full_image.shape[1]))
    count_predictions = np.zeros((classes,full_image.shape[0], full_image.shape[1]))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], full_image.shape[1])
            y2 = min(y1 + tile_size[0], full_image.shape[0])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows
            img = full_image[y1:y2, x1:x2]
            padded_img, padding_index = pad_image(img, tile_size) #only happen in validation or test when the original image size is already smaller than tile_size
            tile_counter += 1
            padded_img = padded_img[None, :, :, :].astype('float32') # extend one dimension
            padded_img = np.transpose(padded_img,(0,3,1,2)) #NCWH

            dl = [[mx.nd.array(padded_img)]]
            data_batch = mx.io.DataBatch(data=dl, label=None,
                                         pad=0, index=nbatch,
                                         provide_data=val_provide_data, provide_label=val_provide_label)
            output_all = predictor.predict(data_batch)
            padded_prediction = output_all[0]["softmax_output"]
            padded_prediction = padded_prediction.asnumpy()
            padded_prediction = np.squeeze(padded_prediction)
            #padded_prediction = predictor(padded_img)[0][0]
            prediction_no_padding = padded_prediction[:,padding_index[0]:padding_index[1],padding_index[2]:padding_index[3]]
            count_predictions[:,y1:y2, x1:x2] += 1
            full_probs[:,y1:y2, x1:x2] += prediction_no_padding  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions #CWH
    return full_probs


def predict_scaler(data, predictor, scales, classes, tile_size, is_densecrf,nbatch,val_provide_data,val_provide_label):
    data = np.squeeze(data) #default batch size is 1

    full_probs = np.zeros((classes, data.shape[0], data.shape[1]))
    h_ori, w_ori = data.shape[0:2]

    for scale in scales:
        scaled_img = cv2.resize(data, (int(scale*w_ori), int(scale*h_ori)))
        scaled_probs = predict_slider(scaled_img, predictor, classes, tile_size,nbatch,val_provide_data,val_provide_label)
        scaled_probs = np.transpose(scaled_probs,(1,2,0))# to HWC
        probs = cv2.resize(scaled_probs, (w_ori,h_ori))
        probs = np.transpose(probs, (2,0,1))
        full_probs += probs
    full_probs /= len(scales)
    if is_densecrf:
        full_probs = dense_crf(full_probs)
    return full_probs





def dense_crf(probs, img=None, n_iters=10,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    h, w, class_num = probs.shape

    probs = probs.transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, class_num)  # Define DenseCRF model.
    U = -np.log(probs)  # Unary potential.
    U = U.reshape((class_num, -1)).astype(np.float32)  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((class_num, h, w)).transpose(1, 2, 0)
    return preds





