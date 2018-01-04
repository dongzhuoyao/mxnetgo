# mxnetgo

[![Build Status](https://travis-ci.org/dongzhuoyao/mxnetgo.svg?branch=master)](https://travis-ci.org/dongzhuoyao/mxnetgo)

a  simple scaffold of MXNet, easy to customize, extremely boosting your research. more examples will be provided.

the target is to make MXNet research as easy as "YiBaSuo".
![yibasuo.jpg](yibasuo.jpg)

# Examples

currently available example: [example/deeplab/README.md](example/deeplab/README.md)

# Characteristics

* no document, because it's very simple to use for anyone who has a basic programming skill.

* a very good starter for research, all you need write only is the network structure.

* modularization for specific task, such as image segmentation, pose estimation etc. 

* reproducible, every experient can produce the same result as the paper indicated.

# Install

Dependencies:

* python2(python3 is current not compatible)

* mxnet==1.0.0

```
pip install -U git+https://github.com/dongzhuoyao/tensorpack.git
# or add `--user` to avoid system-wide installation.
```

## TODO
- [ ] hourglass
- [ ] resnet101 model train
- [ ] deeplab result reproduce
- [ ] DUC
- [ ] PSPNet 
- [ ] Non Local Block
- [ ] large kernel

## Acknowledgement

Some of the example codes are based on others' codes. thanks for their contributions. if this example helps you, please citate their papers.



April, 10
