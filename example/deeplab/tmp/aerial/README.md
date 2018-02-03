
## tensorflow

uploaded model on val result: 87.25%(official:68.7)
uploaded model on val result: 88.7%(official:--)

Val mIoU | Test mIoU
------------ | -------------
87.25 | 68.2
88.7(deeplabv2.naked.aerial.4gpu/model-35385) | ?
deeplabv2res101.aerial(512x512:88.07)|73.41
deeplabv2res101.aerial.4gpu（512x512:88.67)|74.95

## mxnet


Arch|Val mIoU | Test mIoU
------------ | -------------| -------------
deeplabv2res101.4gpu | 321x321:86.67||
deeplabv2res101|321x321:86.91;1025x1025:86.8||
deeplabv2res101.adam.1e-4|87.53||
deeplabv2res101.adam.1e-4.scale4|86.7||
deeplabv2res101.adam.1e-4.scale10|87.5||
deeplabv2res101.adam.1e-4.scale4.freeze|86.7||
deeplabv2res101.adam|epoch1:64.39,terminated||
deeplabv2res101.adam.1e-4.scale15.4gpu|**88.3**||
deeplabv2res101.adam.1e-4.scale10.dcn|87.3||
