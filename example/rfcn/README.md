## Result

|                                 | training data     | testing data | mAP@0.5 | mAP@0.7 | 
|---------------------------------|-------------------|--------------|---------|---------|
| official R-FCN, ResNet-v1-101            | VOC 07+12 trainval| VOC 07 test  | 79.6    | 63.1    | 
| official Deformable R-FCN, ResNet-v1-101 | VOC 07+12 trainval| VOC 07 test  | 82.3    | 67.8    | 
| single-GPU            | VOC 07+12 trainval| VOC 07 test  | 79.0    | 61.87   | 
## dataset prepare

1. Please download COCO and VOC 2007+2012 datasets, and make sure it looks like this:

[VOC2017 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)

 [VOC2017 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

	```
	./data/coco/
	./data/VOCdevkit/VOC2007/
	./data/VOCdevkit/VOC2012/
	```

2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	```
	
	