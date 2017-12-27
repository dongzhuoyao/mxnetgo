# Result in Mxnetgo

## Cityscapes
|                                   | mIoU(official) | mIoU  |
|-----------------------------------|------|-------|
| DeepLab, ResNet-v1-101            | 70.3 | 67.26（new code）??? 可信？ |
| deeplabv2.cs.scale5           | 70.3 | 62.25（new code） |
| deeplabv2.cs.scale8.bs2           | 70.3 | 61.5（new code） |
|deeplabv2.cs.scale1.bs2| 70.3 | ING（new code） |
| Deformable DeepLab, ResNet-v1-101 | 75.2 | 67.2%(new code) |

## Pascal 
|                                   | mIoU(official) | mIoU(old code)  | mIoU(new code)  |
|-----------------------------------|------|-------|------|
| DeepLab, ResNet-v1-101(deeplabv2.pascal:bs2,scale5)            | 70.7 | 69.4 | 67.2 |
|deeplabv2.pascal.bs10.scale2 | 70.7 | 69.4 | 61 |
|deeplabv2.pascal.bs10.scale4| 70.7 | 69.4 | 63.~|
| Deformable DeepLab, ResNet-v1-101 | 75.9 | 74.2 | ? |
|deeplabv2.pascal.dcn.bs10.scale2| 75.9 | 74.2 | 67.0|






# Deformable Convolutional Networks


The major contributors of this repository include [Yuwen Xiong](https://github.com/Orpine), [Haozhi Qi](https://github.com/Oh233), [Guodong Zhang](https://github.com/gd-zhang), [Yi Li](https://github.com/liyi14), [Jifeng Dai](https://github.com/daijifeng001), [Bin Xiao](https://github.com/leoxiaobin), [Han Hu](https://github.com/ancientmooner) and  [Yichen Wei](https://github.com/YichenWei).


## Main Results

|                                 | training data     | testing data | mAP@0.5 | mAP@0.7 | time   |
|---------------------------------|-------------------|--------------|---------|---------|--------|
| R-FCN, ResNet-v1-101            | VOC 07+12 trainval| VOC 07 test  | 79.6    | 63.1    | 0.16s |
| Deformable R-FCN, ResNet-v1-101 | VOC 07+12 trainval| VOC 07 test  | 82.3    | 67.8    | 0.19s |



|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|
| <sub>R-FCN, ResNet-v1-101 </sub>           | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 32.1 | 54.3    |   33.8  | 12.8  | 34.9  | 46.1  | 
| <sub>Deformable R-FCN, ResNet-v1-101</sub> | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 35.7 | 56.8    | 38.3    | 15.2  | 38.8  | 51.5  |
| <sub>Faster R-CNN (2fc), ResNet-v1-101 </sub>           | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 30.3 | 52.1    |   31.4  | 9.9  | 32.2  | 47.4  | 
| <sub>Deformable Faster R-CNN (2fc), </br>ResNet-v1-101</sub> | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 35.0 | 55.0    | 38.3    | 14.3  | 37.7  | 52.0  |



|                                   | training data              | testing data   | mIoU | time  |
|-----------------------------------|----------------------------|----------------|------|-------|
| DeepLab, ResNet-v1-101            | Cityscapes train           | Cityscapes val | 70.3 | 0.51s |
| Deformable DeepLab, ResNet-v1-101 | Cityscapes train           | Cityscapes val | 75.2 | 0.52s |
| DeepLab, ResNet-v1-101            | VOC 12 train (augmented) | VOC 12 val   | 70.7 | 0.08s |
| Deformable DeepLab, ResNet-v1-101 | VOC 12 train (augmented) | VOC 12 val   | 75.9 | 0.08s |


*Running time is counted on a single Maxwell Titan X GPU (mini-batch size is 1 in inference).*

## Requirements: Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet). We tested our code on [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60). Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. We may maintain this repository periodically if MXNet adds important feature in future release.

2. Python 2.7. We recommend using Anaconda2 as it already includes many common packages. We do not support Python 3 yet, if you want to use Python 3 you need to modify the code to make it work.


3. Python packages might missing: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running
	```
	pip install -r requirements.txt
	```
4. For Windows users, Visual Studio 2015 is needed to compile cython module.



## Requirements: Hardware

Any NVIDIA GPUs with at least 4GB memory should be OK.
