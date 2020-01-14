import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha=alpha)


class UNet(chainer.Chain):

    def __init__(self, n_class=2):
        super().__init__()
        with self.init_scope():
            self.n_class = n_class

            self.enco1_1 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
            self.enco1_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)

            self.enco2_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.enco2_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)

            self.enco3_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.enco3_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)

            self.enco4_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.enco4_2 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)

            self.enco5_1 = L.Convolution2D(None,1012, ksize=3, stride=1, pad=1)

            self.deco6_1 = L.Convolution2D(None,1012, ksize=3, stride=1, pad=1)
            self.deco6_2 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)

            self.deco7_1 = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1)
            self.deco7_2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)

            self.deco8_1 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.deco8_2 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)

            self.deco9_1 = L.Convolution2D(None, 128, ksize=3, stride=1, pad=1)
            self.deco9_2 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)
            self.deco9_3 = L.Convolution2D(None,  64, ksize=3, stride=1, pad=1)

            self.final_layer = L.Convolution2D(None, n_class, ksize=1)

            self.bn1_1 = L.BatchNormalization(  64)
            self.bn1_2 = L.BatchNormalization(  64)

            self.bn2_1 = L.BatchNormalization( 128)
            self.bn2_2 = L.BatchNormalization( 128)

            self.bn3_1 = L.BatchNormalization( 256)
            self.bn3_2 = L.BatchNormalization( 256)

            self.bn4_1 = L.BatchNormalization( 512)
            self.bn4_2 = L.BatchNormalization( 512)

            self.bn5_1 = L.BatchNormalization(1012)

            self.bn6_1 = L.BatchNormalization(1012)
            self.bn6_2 = L.BatchNormalization( 512)

            self.bn7_1 = L.BatchNormalization( 512)
            self.bn7_2 = L.BatchNormalization( 256)

            self.bn8_1 = L.BatchNormalization( 256)
            self.bn8_2 = L.BatchNormalization( 128)

            self.bn9_1 = L.BatchNormalization( 128)
            self.bn9_2 = L.BatchNormalization(  64)
            self.bn9_3 = L.BatchNormalization(  64)

    # def __call__(self, x): #x = (batchsize, 3, 360, 480)
    def _logits(self, x):
        h1_1 = F.relu(self.bn1_1(self.enco1_1(x)))
        h1_2 = F.relu(self.bn1_2(self.enco1_2(h1_1)))

        pool1 = F.max_pooling_2d(h1_2, 2, stride=2, return_indices=False) #(batchsize,  64, 180, 240)

        h2_1 = F.relu(self.bn2_1(self.enco2_1(pool1)))
        h2_2 = F.relu(self.bn2_2(self.enco2_2(h2_1)))
        pool2 = F.max_pooling_2d(h2_2, 2, stride=2, return_indices=False) #(batchsize, 128,  90, 120)

        h3_1 = F.relu(self.bn3_1(self.enco3_1(pool2)))
        h3_2 = F.relu(self.bn3_2(self.enco3_2(h3_1)))
        pool3 = F.max_pooling_2d(h3_2, 2, stride=2, return_indices=False) #(batchsize, 256,  45,  60)

        h4_1 = F.relu(self.bn4_1(self.enco4_1(pool3)))
        h4_2 = F.relu(self.bn4_2(self.enco4_2(h4_1)))
        pool4 = F.max_pooling_2d(h4_2, 2, stride=2, return_indices=False) #(batchsize, 256,  23,  30)

        h5_1 = F.relu(self.bn5_1(self.enco5_1(pool4)))

        up5 = F.unpooling_2d(h5_1, ksize=2, stride=2, outsize=(pool3.shape[2], pool3.shape[3]))
        h6_1 = F.relu(self.bn6_1(self.deco6_1(F.concat((up5, h4_2)))))
        h6_2 = F.relu(self.bn6_2(self.deco6_2(h6_1)))

        up6 = F.unpooling_2d(h6_2, ksize=2, stride=2, outsize=(pool2.shape[2], pool2.shape[3]))
        h7_1 = F.relu(self.bn7_1(self.deco7_1(F.concat((up6, h3_2)))))
        h7_2 = F.relu(self.bn7_2(self.deco7_2(h7_1)))

        up7 = F.unpooling_2d(h7_2, ksize=2, stride=2, outsize=(pool1.shape[2], pool1.shape[3]))
        h8_1 = F.relu(self.bn8_1(self.deco8_1(F.concat((up7, h2_2)))))
        h8_2 = F.relu(self.bn8_2(self.deco8_2(h8_1)))

        up8 = F.unpooling_2d(h8_2, ksize=2, stride=2, outsize=(x.shape[2], x.shape[3])) #x = (batchsize, 128, 360, 480)
        h9_1 = F.relu(self.bn9_1(self.deco9_1(F.concat((up8, h1_2)))))
        h9_2 = F.relu(self.bn9_2(self.deco9_2(h9_1)))
        h9_3 = F.relu(self.bn9_3(self.deco9_3(h9_2)))

        h = self.final_layer(h9_3)

        return h

    def __call__(self, x, t):
        xp = cuda.get_array_module(x)
        h = self._logits(x)

        """
        if chainer.config.train:
            class_weight = self.class_weight / self.class_weight.sum()
        else:
            class_weight = xp.array([1., 1.], dtype=self.class_weight.dtype)
        """

        loss = F.softmax_cross_entropy(h, t)
        #with chainer.using_config('use_cudnn', 'never'):
        #    # loss = F.softmax_cross_entropy(h, t, class_weight=class_weight)
        #    loss = F.softmax_cross_entropy(h, t)
        #if xp.isnan(loss.data):
        #    raise RuntimeError("ERROR in MyFcn: loss.data is nan!")
        if xp.isnan(loss.data):
            raise RuntimeError("ERROR in UNet: loss.data is nan!")

        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'accuracy': accuracy,
            'error_rate': 1 - accuracy,
        }, self)
        """
        chainer.report(
            {'class_weight[{}]'.format(i): w
             for i, w in enumerate(cuda.to_cpu(class_weight).ravel())}, self)
        """
        return loss

    def predict(self, x):
        with chainer.using_config('train', False):
            h = self._logits(x)
            #with chainer.using_config('use_cudnn', 'never'):
            #    return F.softmax(h)
            return F.softmax(h)


def _main():
    pass


if __name__ == '__main__':
    _main()
