# Result in Mxnetgo

Deeplab Paper result: MSC+Coco+Aug+ASPP+CRF=77.69% mIoU

## Pascal 
|                                   | mIoU(official,8GPU) |  mIoU(my)|
|-----------------------------------|------|------|
|deeplabv2.pascal| 70.7 |**70.45**(without Coco,ASPP,CRF)|
|deeplabv2.pascal.4gpu.scale4|70.7|70.99|
|deeplabv2.pascal.4gpu.scale16|70.7|**71.7**|
|deeplabv2.pascal.dcn| 75.9 |  **74.7**|



## Cityscapes

Deeplab Paper result: Full+Aug+ASPP+CRF=71.4% mIoU

|                                   | mIoU(official,8GPU) | mIoU(my)  |
|-----------------------------------|------|-------|
|deeplabv2.cs.scale4.officiallr.full.longer|70.3|**69.4**|
| Deformable DeepLab, ResNet-v1-101 | 75.2 |-- |


more experimental result can be seen in [NOTE](tmp/NOTE.md)


## some notes

* lr schedule is very important, in most tensorflow framework, the lr schedule is [(3, 1e-4), (5, 1e-5), (7, 8e-6)], however, in mxnet, as paper indicated. the lr schedule is [(4, 1e-3), (6, 1e-4)](total 30k iterations).

* large image size can bring about  nearly 1% gain in image segmentation.

* how much gain dilation convolution  can bring need more experiments to explore.
