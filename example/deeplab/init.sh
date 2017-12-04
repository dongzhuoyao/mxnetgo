#!/usr/bin/env bash

cd lib/bbox
python setup_linux.py build_ext --inplace
cd ../dataset/pycocotools
python setup_linux.py build_ext --inplace
cd ../../nms
python setup_linux.py build_ext --inplace
cd ../..