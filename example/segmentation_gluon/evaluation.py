import mxnet as mx
from mxnet import gluon
import augment
import cfg
from data.cityscapes import CityScapes
from model.lkm import LKM
from model.deeplab import DeepLab
import argparse
import os


def parse_arg():
    parser = argparse.ArgumentParser(
        description='evaluate segmentation CNN with gluon')
    parser.add_argument(
        '--name',
        help='name of the network',
        dest='name',
        type=str,
        default='DeepLab_512_cityscapes')
    parser.add_argument(
        '--num_workers',
        help='number of workers to load data',
        dest='num_workers',
        type=int,
        default=4)
    parser.add_argument(
        '--batch_size',
        help='batch size',
        dest='batch_size',
        type=int,
        default=16)
    args = parser.parse_args()
    return args


nets = {'LKM': LKM, 'DeepLab': DeepLab}
args = parse_arg()


def accuracy_eval(net, ctx, loader):
    acc_metric = mx.metric.Accuracy()
    with mx.autograd.predict_mode():
        for img, lbl in loader:
            img = img.as_in_context(ctx)
            lbl = lbl.as_in_context(ctx)
            pred = net(img)
            pred = mx.nd.argmax(pred, axis=1)
            acc_metric.update(preds=pred, labels=lbl)
    return acc_metric.get()[1]


def eval_main():
    dataset = CityScapes(cfg.cityscapes_root, 'val', augment.cityscapes_val)
    loader = gluon.data.DataLoader(
        dataset, args.batch_size, False, num_workers=args.num_workders)

    save_root = os.path.join('save', args.name)
    net = nets[args.name.split('_')[0]](pretrained=False)
    net.load_params(os.path.join(save_root, 'weights'), ctx=cfg.ctx)
    net.hybridize()

    acc = accuracy_eval(net, cfg.ctx, loader)
    print('{} : {:.6f}%'.format(args.name, acc))


if __name__ == '__main__':
    eval_main()
