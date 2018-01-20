# Result in Mxnetgo

Deeplab Paper result: MSC+Coco+Aug+ASPP+CRF=77.69% mIoU

## Pascal 
|                                   | mIoU(official,8GPU) |  mIoU(my)|
|-----------------------------------|------|------|
|deeplabv2.pascal| 70.7 |**70.45**(without Coco,ASPP,CRF)|
|deeplabv2.pascal.4gpu.scale16|70.7|**71.7**|
|deeplabv2.pascal.dcn| 75.9 |  **74.7**|



## Cityscapes

Deeplab Paper result: Full+Aug+ASPP+CRF=71.4% mIoU

|                                   | mIoU(official,8GPU) | mIoU(my)  |
|-----------------------------------|------|-------|
|deeplabv2.cs.scale4.officiallr.full.longer|70.3|**69.4**|
| Deformable DeepLab, ResNet-v1-101 | 75.2 |-- |


more experimental result can be seen in [NOTE](tmp/NOTE.md)

## Doubt

* why tensorpack dataload speed is about 1.2items/s, while mxnetgo is only 0.3item/s 


## some notes

* lr schedule is very important, in most tensorflow framework, the lr schedule is [(3, 1e-4), (5, 1e-5), (7, 8e-6)], however, in mxnet, as paper indicated. the lr schedule is [(4, 1e-3), (6, 1e-4)](total 30k iterations).

* large image size can bring about  nearly 1% gain in image segmentation.

* how much gain dilation convolution  can bring need more experiments to explore.

* RandomResize image with a nearest-neighbor interpolation can slightly increase mIoU by 0.8%???

* iteration number is important or total image numbers?

* To stable the statistics of the BatchNormormalization, the combination of the image size and batch size should tuned accurately as indicated in [InPlace-ABN](https://arxiv.org/abs/1712.02616)

![misc/bs-is.jpg](misc/bs-is.jpg)

here some other method' choice as follows:

|      Method                             | image crop size | batch size(single gpu)  |
|-----------------------------------|------|-------|
|deeplabv2-resnet101|321|10|
|PSPNet_VOC2012-resnet101|473|as max as possible|
|PSPNet_Cityscapes-resnet101|713|as max as possible|
|Inplace_ABN_Cityscapes-resnext101|672|16|
|Inplace_ABN_CocoStuff-resnext101|600|16|
|DeformableConvolutionNetworks-Deeplab-Cityscapes| 768*1024|1|
|DeformableConvolutionNetworks-Deeplab-PascalVOC| 768*1024|1|





