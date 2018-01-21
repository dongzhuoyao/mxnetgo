import argparse


def parse_arg():
    parser = argparse.ArgumentParser(
        description='training segmentation CNN with gluon')

    # network name
    parser.add_argument(
        '--name',
        help='name of the network',
        dest='name',
        type=str,
        default='LKM_512_cityscapes')
    # use dropbox
    parser.add_argument(
        '--dropbox',
        help='copy save files to dropbox',
        dest='dropbox',
        action='store_true')
    # learning rate
    parser.add_argument(
        '--lr', help='learning rate', dest='lr', type=float, default=0.00025)
    parser.add_argument(
        '--lr_decay_factor',
        help='learning rate decay factor',
        dest='lr_decay_factor',
        type=float,
        default=0.1)
    parser.add_argument(
        '--lr_decay_steps',
        nargs='*',
        help='learning rate decay step',
        dest='lr_decay_step',
        type=int,
        default=[40, 80])
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
        default=8)
    # num epoch
    parser.add_argument(
        '--num_epoch',
        help='number of epoch',
        dest='num_epoch',
        type=int,
        default=120)
    # checkpoint
    parser.add_argument(
        '--checkpoint',
        help='period of epochs to checkpoint',
        dest='checkpoint',
        type=int,
        default=1)

    args = parser.parse_args()
    return args


args = parse_arg()

if __name__ == '__main__':
    print(args)
