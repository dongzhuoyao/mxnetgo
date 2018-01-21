import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import cfg


class GlobalConvolutionalNetwork(nn.HybridBlock):
    def __init__(self, channels, k):
        super(GlobalConvolutionalNetwork, self).__init__()
        pad = k // 2
        with self.name_scope():
            self.left = nn.HybridSequential()
            self.left.add(
                nn.Conv2D(channels, kernel_size=(k, 1), padding=(pad, 0)),
                nn.Conv2D(channels, kernel_size=(1, k), padding=(0, pad)))
            self.right = nn.HybridSequential()
            self.right.add(
                nn.Conv2D(channels, kernel_size=(1, k), padding=(0, pad)),
                nn.Conv2D(channels, kernel_size=(k, 1), padding=(pad, 0)))

    def hybrid_forward(self, F, x):
        return self.left(x) + self.right(x)


class BoundaryRefineModule(nn.HybridBlock):
    def __init__(self, channels):
        super(BoundaryRefineModule, self).__init__()
        with self.name_scope():
            self.layer = nn.HybridSequential()
            self.layer.add(
                nn.Conv2D(channels, kernel_size=3, padding=1),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(channels, kernel_size=3, padding=1))

    def hybrid_forward(self, F, x):
        return x + self.layer(x)


class LKM(gluon.HybridBlock):
    def __init__(self, pretrained=False, large_kernel=15, base_prefix=''):
        super(LKM, self).__init__()
        with self.name_scope():
            resnet = gluon.model_zoo.vision.resnet101_v1(
                pretrained=pretrained, prefix=base_prefix).features
            self.layer0 = nn.HybridSequential()
            self.layer0.add(*resnet[0:4])
            self.layer1 = resnet[4]
            self.layer2 = resnet[5]
            self.layer3 = resnet[6]
            self.layer4 = resnet[7]

            self.gcn_4 = GlobalConvolutionalNetwork(cfg.n_class, large_kernel)
            self.gcn_8 = GlobalConvolutionalNetwork(cfg.n_class, large_kernel)
            self.gcn_16 = GlobalConvolutionalNetwork(cfg.n_class, large_kernel)
            self.gcn_32 = GlobalConvolutionalNetwork(cfg.n_class, large_kernel)

            self.br_1 = BoundaryRefineModule(cfg.n_class)
            self.br_2 = BoundaryRefineModule(cfg.n_class)
            self.br_4_1 = BoundaryRefineModule(cfg.n_class)
            self.br_4_2 = BoundaryRefineModule(cfg.n_class)
            self.br_8_1 = BoundaryRefineModule(cfg.n_class)
            self.br_8_2 = BoundaryRefineModule(cfg.n_class)
            self.br_16_1 = BoundaryRefineModule(cfg.n_class)
            self.br_16_2 = BoundaryRefineModule(cfg.n_class)
            self.br_32 = BoundaryRefineModule(cfg.n_class)

            self.deconv_2 = nn.Conv2DTranspose(
                cfg.n_class, kernel_size=4, strides=2, padding=1)
            self.deconv_4 = nn.Conv2DTranspose(
                cfg.n_class, kernel_size=4, strides=2, padding=1)
            self.deconv_8 = nn.Conv2DTranspose(
                cfg.n_class, kernel_size=4, strides=2, padding=1)
            self.deconv_16 = nn.Conv2DTranspose(
                cfg.n_class, kernel_size=4, strides=2, padding=1)
            self.deconv_32 = nn.Conv2DTranspose(
                cfg.n_class, kernel_size=4, strides=2, padding=1)

    def hybrid_forward(self, F, x):
        x = self.layer0(x)  # 1 / 2
        x = self.layer1(x)  # 1 / 4

        score4 = self.br_4_1(self.gcn_4(x))

        x = self.layer2(x)  # 1 / 8

        score8 = self.br_8_1(self.gcn_8(x))

        x = self.layer3(x)  # 1 / 16

        score16 = self.br_16_1(self.gcn_16(x))

        x = self.layer4(x)  # 1 / 32

        score32 = self.br_32(self.gcn_32(x))

        score16 = self.br_16_2(score16 + self.deconv_32(score32))
        score8 = self.br_8_2(score8 + self.deconv_16(score16))
        score4 = self.br_4_2(score4 + self.deconv_8(score8))

        return self.br_1(self.deconv_2(self.br_2(self.deconv_4(score4))))


def lkm_test():
    net = LKM(pretrained=True)
    net.collect_params().initialize(
        init=mx.initializer.Xavier(magnitude=2.0), ctx=cfg.ctx)
    x = nd.random_uniform(shape=(2, 3, 512, 512))
    y = net(x)
    print(y)


if __name__ == '__main__':
    lkm_test()
