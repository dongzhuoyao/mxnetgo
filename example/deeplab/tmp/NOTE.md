# Result in Mxnetgo

## Cityscapes
|                                   | mIoU(official) | mIoU  |
|-----------------------------------|------|-------|
| DeepLab, ResNet-v1-101            | 70.3 | -- |
| deeplabv2.cs.scale5           | 70.3 | 62.25（new code） |
| deeplabv2.cs.scale8.bs2           | 70.3 | 61.5（new code） |
| deeplabv2.cs.scale4.bs2.newlr.sgd| 70.3| 62.6|
|deeplabv2.cs.scale4.bs2.officiallr| 70.3|67.18(new code)
|deeplabv2.cs.scale4.officiallr.full|70.3| 68.95|
|deeplabv2.cs.scale4.officiallr.full.longer|70.3|**69.4**|
|deeplabv2.cs.scale1.bs2| 70.3 | 50.~（new code） |
|deeplabv2.cs.imagesize672(size:672,epoch_scale=4)|70.3|65|
| Deformable DeepLab, ResNet-v1-101 | 75.2 |-- |


## Pascal 
|                                   | mIoU(official) | mIoU(old code)  | mIoU(new code)  |
|-----------------------------------|------|-------|------|
| DeepLab, ResNet-v1-101(deeplabv2.pascal:bs2,scale5)            | 70.7 | 69.4 | 67.2 |
|deeplabv2.pascal.bs10.scale2 | 70.7 | 69.4 | 61 |
|deeplabv2.pascal.bs10.scale4| 70.7 | 69.4 | 63.~|
|deeplabv2.cs.bs2.github| 70.7|--| 65|
|deeplabv2.pascal.bs10.scale4.newlr.sgd| 70.7 | 69.4 | 66.9(msf:67.9)|
|deeplabv2.pascal.bs10.scale4.officiallr(MSF)| 70.7 | 69.4 |**70.45**|
|deeplabv2.pascal.4gpu|--|--|70.5|
|deeplabv2.pascal.4gpu.scale4--|--|70.99|
|deeplabv2.pascal.4gpu.scale16|--|--|71.7|
| Deformable DeepLab, ResNet-v1-101 | 75.9 | 74.2 | ? |
|deeplabv2.pascal.dcn| 75.9 | 74.2 |  **74.7**|


## some notes

lr schedule is very important, in most tensorflow framework, the lr schedule is [(3, 1e-4), (5, 1e-5), (7, 8e-6)], however, in mxnet, as paper indicated. the lr schedule is [(4, 1e-3), (6, 1e-4)](total 30k iterations).


