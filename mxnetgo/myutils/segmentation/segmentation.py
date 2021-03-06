#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: segmentation.py
# Author: Tao Hu <taohu620@gmail.com>

import numpy as np
from math import ceil
import cv2,colorsys
import mxnet as mx
from mxnetgo.myutils import logger
import os, sys
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


def update_confusion_matrix(pred, label, conf_m, nb_classes, ignore=255):
    ignore_index = label != ignore
    seg_gt = label[ignore_index].astype('int32')
    seg_pred = pred[ignore_index].astype('int32')
    index = (seg_gt * nb_classes + seg_pred).astype('int32')
    label_count = np.bincount(index)
    for i_label in range(nb_classes):
        for i_pred_label in range(nb_classes):
            cur_index = i_label * nb_classes + i_pred_label
            if cur_index < len(label_count):
                conf_m[i_label, i_pred_label] += label_count[cur_index]  # notice here, first dimension is label,second dimension is prediction.
    return conf_m

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = max(target_size[0] - img.shape[0], 0)
    cols_missing = max(target_size[1] - img.shape[1], 0)
    try:
        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    except  Exception as e:
        print(str(e))
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

    img_color[label == 255] = [255,255,255]
    return img_color


def predict_slider(full_image, predictor, classes, tile_size):
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

            data = [[mx.nd.array(padded_img)]]

            output_all = predictor(data)
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


def predict_scaler(data, predictor, scales, classes, tile_size, is_densecrf):
    data = np.squeeze(data).astype(np.int16) #default batch size is 1

    full_probs = np.zeros((classes, data.shape[0], data.shape[1]))
    h_ori, w_ori = data.shape[0:2]

    for scale in scales:
        scaled_img = cv2.resize(data, (int(scale*w_ori), int(scale*h_ori)))
        scaled_probs = predict_slider(scaled_img, predictor, classes, tile_size)
        scaled_probs = np.transpose(scaled_probs,(1,2,0))# to HWC
        probs = cv2.resize(scaled_probs, (w_ori,h_ori))
        probs = np.transpose(probs, (2,0,1))
        full_probs += probs
    full_probs /= len(scales)
    #if is_densecrf:
    #    full_probs = dense_crf(full_probs)
    return full_probs


if __name__ == '__main__':
    label = np.array([1,1,1,0,0,0,0])
    pred = np.array([0,0,0,0,0,0,0])
    cm = np.array([[0,0],[0,0]])
    cm = update_confusion_matrix(pred,label,cm,2)
    pass









