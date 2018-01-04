# Result in Mxnetgo

## Pascal 
|                                   | mIoU(official) |  mIoU|
|-----------------------------------|------|------|
|deeplabv2.pascal| 70.7 |**70.45**|
|deeplabv2.pascal.4gpu.scale16|--|**71.7**|
|deeplabv2.pascal.dcn| 75.9 |  **74.7**|


## Cityscapes
|                                   | mIoU(official) | mIoU  |
|-----------------------------------|------|-------|
|deeplabv2.cs|70.3|69.4|
| Deformable DeepLab, ResNet-v1-101 | 75.2 |-- |


more experimental result can be seen in [tmp/NOTE.md](tmp/NOTE.md)


## some notes

lr schedule is very important, in most tensorflow framework, the lr schedule is [(3, 1e-4), (5, 1e-5), (7, 8e-6)], however, in mxnet, as paper indicated. the lr schedule is [(4, 1e-3), (6, 1e-4)](total 30k iterations).



