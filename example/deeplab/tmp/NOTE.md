# Result in Mxnetgo

## ADE20K

|                                   | mIoU,acc(official) | mIoU(my)  |
|-----------------------------------|------|-------|
|PSPNet,ResNet50+DA+AL+PSP+MS,validation set(from PSPNet github)|42.78/80.76|?|
|PSPNet,ResNet269+DA+AL+PSP+MS,validation set|44.94/81.69|?|

## Cityscapes
|                                   | mIoU(official) | mIoU  |
|-----------------------------------|------|-------|
|PSPNet,ResNet101,fine set,test result|78.4||
|PSPNet,ResNet101,fine+coarse set,test result|80.2||
|-----------------------------------|------|-------|
| DeepLab, ResNet-v1-101            | 70.3 | -- |
| deeplabv2.cs.scale5           | 70.3 | 62.25（new code） |
| deeplabv2.cs.scale8.bs2           | 70.3 | 61.5（new code） |
| deeplabv2.cs.scale4.bs2.newlr.sgd| 70.3| 62.6|
|deeplabv2.cs.scale4.bs2.officiallr| 70.3|67.18(new code)
|deeplabv2.cs.scale4.officiallr.full|70.3| 68.95|
|deeplabv2.cs.scale4.officiallr.full.longer|70.3|**69.4**|
|deeplabv2.cs.scale1.bs2| 70.3 | 50.~（new code） |
|deeplabv2.cs.imagesize672(size:672,epoch_scale=4)|70.3|65,because epoch_scale is too small, it should be 18, however, the dataload speed is too slow in mxnetgo|
|deeplabv1res101.cs.imagesize672.scale18.adam|70.3|44.99 in epoch 6, stopped because of OOM|
|deeplabv1res101.cs.imagesize672.scale18|70.3|48.4|
| Deformable DeepLab, ResNet-v1-101 | 75.2 |-- |


## Pascal 
|                                   | mIoU(official) | mIoU|
|-----------------------------------|------|------|
PSPNet,ResNet101, test result|82.6||
PSPNet,ResNet101,,coco pretrain, test result|85.4||
|Deeplabv3,without coco pretrain, val result|79.77||
|-----------------------------------|------|------|
| DeepLab, ResNet-v1-101(deeplabv2.pascal:bs2,scale5)| 70.7 | 67.2 |
|deeplabv2.pascal.bs10.scale2 | 70.7  | 61 |
|deeplabv2.pascal.bs10.scale4| 70.7| 63.~|
|deeplabv2.cs.bs2.github| 70.7| 65|
|deeplabv2.pascal.bs10.scale4.newlr.sgd| 70.7  | 66.9(msf:67.9)|
|deeplabv2.pascal.bs10.scale4.officiallr(MSF)| 70.7 |**70.45**|
deeplabv2res101.pascal.imagesize473.scale8.unfreeze|70.7|67|
|deeplabv1res101.pascal.imagesize473.scale8|70.7|69.3|
|deeplabv2.pascal.4gpu|--|70.5|
|deeplabv2.pascal.4gpu.scale4|.|70.99|
|deeplabv2.pascal.4gpu.scale16|--|71.7|
| Deformable DeepLab, ResNet-v1-101 | 75.9 | 74.2 |
|deeplabv2.pascal.dcn| 75.9 |  **74.7**|


|                     Method      | mIoU| my|
|-----------------------------------|------|-------|
|new.deeplabv2res101.pascal.imagesize473.scale8|70.7|59.64|
|newmodel.deeplabv2res101.pascal.imagesize473.scale4.lr1e-4|70.7|56|
|newmodel.deeplabv2res101.pascal.imagesize473.scale4.freeze.adam|70.7|4 in epoch1,2,3, stopped|
|newmodel.deeplabv2res101.pascal.imagesize473.scale4.freeze|70.7|59.5|
|newmodel.deeplabv2res101.pascal.imagesize473.scale8.freeze|70.7|60,second time **73.3**|
newmodel.deeplabv2res101aspp.pascal.imagesize473.scale8.freeze|:question:||
|newmodel.deeplabv2res101.pascal.imagesize473.scale12|70.7|57.7% in epoch 7,stucked|
|newmodel.deeplabv2res101.pascal.imagesize473.scale4|70.7|60.3
|newmodel.deeplabv2res101.pascal.imagesize473.scale8.bnfix.dilationfix||epoch7:72,got stuck,second time: **73.2**|
newmodel.deeplabv2res101.pascal.imagesize473.scale8.bnfix.dilationfix.adam1e-4||:question:|
newmodel.deeplabv2res101.pascal.imagesize473.scale16.bnfix.dilationfix.4gpu||**74.1**|
newmodel.deeplabv2res101.pascal.imagesize473.scale8.freeze.4gpu||74.3|
|newmodel.deeplabv2res101.cs.imagesize672.scale18||stopped, no use|
newmodel.deeplabv2res101.cs.imagesize673.scale9.bnfix.dilationfix||53.4|
|-----------------------------------|--|--|
newmodel.deeplabv1res101.pascal.imagesize473.scale4.freezestage123|70.7|56.2
newmodel.deeplabv1res101.pascal.imagesize473.scale4.freezestage12|70.7|62
newmodel.deeplabv1res101.pascal.imagesize473.scale4.freeze|70.7|52.19|
newmodel.deeplabv1res101.pascal.imagesize473.scale8.freeze(second time)(fix beta,gamma;use_global_stats = not is_train)|70.7|**73.7** :smile:|
newmodel.deeplabv1res101.pascal.imagesize473.scale4|70.7|63.9|
newmodel.deeplabv1res101.pascal.imagesize473.scale8.freezestage1|70.7|55.31|
newmodel.deeplabv1res101.pascal.imagesize473.scale8.bnfix|70.7|68.65|
|-----------------------------------|--|--|
newmodel.deeplabv1res101.cs.imagesize673.scale18.freeze||53.26,second time: OOM|
newmodel.deeplabv1res101.cs.imagesize673.scale18.freeze.adam||epoch1:11.46,terminated|
newmodel.deeplabv1res101.cs.imagesize673.scale18.freeze.adam1e-4||:question:|
newmodel.deeplabv1res101.cs.imagesize673.scale18.freeze.adam1e-2||epoch1:3.3,terminated|
|-----------------------------------|--|--|
resnext.deeplabv1res101.pascal.imagesize473.scale8.bnfix.dilationfix||70.1|
resnext.deeplabv1res101.pascal.imagesize473.scale8.bnfix.dilationfix.freeze||**71.8**|

## some notes

lr schedule is very important, in most tensorflow framework, the lr schedule is [(3, 1e-4), (5, 1e-5), (7, 8e-6)], however, in mxnet, as paper indicated. the lr schedule is [(4, 1e-3), (6, 1e-4)](total 30k iterations).


