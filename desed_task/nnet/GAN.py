import torch.nn as nn
import torch


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class GAN(nn.Module):
    def __init__(
            self,
            activation="Relu",
            conv_dropout=0,
            gan_init_channel=64,
            gan_latent_dim=1560,
            gan_kernel_size=[3, 3, 3, 3, 3],
            gan_padding=[1, 1, 1, 1, (2,1)],
            gan_stride=[1, 1, 1, 1, 1],
            gan_nb_filters=[ 512, 256, 128, 64, 1],
            gan_upsampling=[(1, 2), (1, 2), (2, 2), (2, 2)],
            normalization="batch",
            **transformer_kwargs
    ):
        """
            Initialization of GAN network s

        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(GAN, self).__init__()

        self.nb_filters = gan_nb_filters
        self.gan_init_channel = gan_init_channel
        self.l1 = nn.Sequential(nn.Linear(gan_latent_dim, gan_init_channel * 156 * 8))
        cnn = nn.Sequential()

        def conv_block(i, normalization="batch", dropout=None, activ="relu"):
            nIn = self.gan_init_channel if i == 0 else gan_nb_filters[i - 1]
            nOut = gan_nb_filters[i]

            # 【1】 Normalization
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nIn, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nIn))

            # 【2】 Convolution
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, gan_kernel_size[i], gan_stride[i], gan_padding[i]),
            )
            # 【3】 activation
            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))
            # 【4】Dropout
            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        last = len(gan_nb_filters) - 1
        for i in range(len(gan_nb_filters)):
            conv_block(i, normalization=normalization, dropout=conv_dropout, activ=activation)

            if i < last:
                cnn.add_module(
                    "upsampling{0}".format(i), nn.Upsample(scale_factor=gan_upsampling[i])
                )  # bs x tframe x mels


        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # generate mel spectrogram
        # x = x.squeeze(-1)
        out = self.l1(x)
        out = out.view(out.shape[0], 64, 156, 8)
        mel_spec = self.cnn(out)
        mel_spec = mel_spec.transpose(2, 3).squeeze(1)
        return mel_spec


if __name__ == "__main__":
    from torchsummary import summary

    model = GAN()

    # summary(model, input_size=(1560,1), device='cpu')

    test_embeddings = torch.randn(10, 1560)
    print(test_embeddings.shape)
    res = model(test_embeddings)
    print(res.shape)
