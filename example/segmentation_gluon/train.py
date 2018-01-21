import os
import pickle
from subprocess import call

import mxnet as mx
import mxnet.autograd as ag
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import augment
import cfg
from data.cityscapes import CityScapes
from model.lkm import LKM
from model.deeplab import DeepLab
import time
import argparse
import math


def parse_arg():
    parser = argparse.ArgumentParser(
        description='training segmentation CNN with gluon')

    # network name
    parser.add_argument(
        '--name',
        help='name of the network',
        dest='name',
        type=str,
        default='DeepLab_512_cityscapes')
    # use dropbox
    parser.add_argument(
        '--dropbox',
        help='copy save files to dropbox',
        dest='dropbox',
        action='store_true')
    # learning rate
    parser.add_argument(
        '--lr', help='learning rate', dest='lr', type=float, default=0.001)
    # weight decay
    parser.add_argument(
        '--weight_decay',
        help='weight decay',
        dest='wd',
        type=float,
        default=0.0001)
    # batch size
    parser.add_argument(
        '--batch_size',
        help='batch size',
        dest='batch_size',
        type=int,
        default=6)
    # num epoch
    parser.add_argument(
        '--num_epoch',
        help='number of epoch',
        dest='num_epoch',
        type=int,
        default=100)
    # checkpoint
    parser.add_argument(
        '--num_workers',
        help='number of workers to load data',
        dest='num_workers',
        type=int,
        default=4)
    parser.add_argument(
        '--checkpoint',
        help='period of epochs to checkpoint',
        dest='checkpoint',
        type=int,
        default=1)

    args = parser.parse_args()
    return args


nets = {'LKM': LKM, 'DeepLab': DeepLab}
args = parse_arg()


def train(name, train_loader, ctx, load_checkpoint, learning_rate, num_epochs,
          weight_decay, checkpoint):
    records = {'losses': []}
    criterion = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    criterion.hybridize()

    save_root = os.path.join('save', name)

    if not os.path.exists(save_root):
        call(['mkdir', '-p', save_root])
    with open(os.path.join(save_root, args.name + '_hp'), 'wb') as f:
        pickle.dump(args, f)

    loaded = False
    if load_checkpoint:
        save_files = set(os.listdir(save_root))
        if {'weights', 'trainer', 'records'} <= save_files:
            print('Loading checkpoint')
            net = nets[name.split('_')[0]](pretrained=False)
            net.load_params(os.path.join(save_root, 'weights'), ctx=cfg.ctx)
            trainer = gluon.Trainer(net.collect_params(), 'sgd', {
                'learning_rate': learning_rate,
                'momentum': 0.9,
                'wd': weight_decay,
            })
            trainer.load_states(os.path.join(save_root, 'trainer'))
            with open(os.path.join(save_root, 'records'), 'rb') as f:
                records = pickle.load(f)
            loaded = True
        else:
            print('Checkpoint files don\'t exist.')
            print('Skip loading checkpoint')

    if not loaded:
        net = nets[name.split('_')[0]](pretrained=True)
        net.collect_params().initialize(
            mx.initializer.MSRAPrelu(), ctx=cfg.ctx)
        net.collect_params().reset_ctx(cfg.ctx)
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {
            'learning_rate': learning_rate,
            'momentum': 0.9,
            'wd': weight_decay,
        })

    net.hybridize()

    print('Start training')
    last_epoch = len(records['losses']) - 1

    for epoch in range(last_epoch + 1, num_epochs):

        trainer.set_learning_rate(
            args.lr * math.pow(1 - epoch / num_epochs, 0.9))
        iter_count = 0
        t0 = time.time()
        running_loss = 0.0

        for data, label in train_loader:

            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            batch_size = data.shape[0]
            with ag.record(train_mode=True):
                output = net(data)
                loss = criterion(output, label)
            loss.backward()
            trainer.step(batch_size)

            _loss = nd.sum(loss).asscalar() / batch_size
            print(
                '\rEpoch {} LR {:.6f} Iter {} Loss {:.4f} '.format(
                    epoch, trainer.learning_rate, iter_count, _loss),
                end='')
            iter_count += 1

            running_loss += _loss

        t1 = time.time()
        print('\rEpoch {} : Loss {:.4f}  Time {:.2f}min'.format(
            epoch, running_loss, (t1 - t0) / 60))
        records['losses'].append(running_loss)

        if (epoch + 1) % checkpoint == 0:
            print('\rSaving checkpoint', end='')
            net.save_params(os.path.join(save_root, 'weights'))
            trainer.save_states(os.path.join(save_root, 'trainer'))
            with open(os.path.join(save_root, 'records'), 'wb') as f:
                pickle.dump(records, f)
            if args.dropbox:
                call(
                    ['cp', '-r', save_root,
                     os.path.join(cfg.home, 'Dropbox')])
            print('\rFinish saving checkpoint', end='')
    print('\nFinish training')


def main():
    train_dataset = CityScapes(cfg.cityscapes_root, 'train',
                               augment.cityscapes_train)
    train_loader = gluon.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    train(args.name, train_loader, cfg.ctx, True, args.lr, args.num_epoch,
          args.wd, args.checkpoint)


if __name__ == '__main__':
    main()
