import warnings

import torch.nn as nn
import torch
from .GAN import GAN
from .CRNN import CRNN


class CRNN_GAN(nn.Module):
    def __init__(
        self,
        config
    ):
        super(CRNN_GAN, self).__init__()
        self.crnn = CRNN(**config["net"])
        self.gan = GAN(**config["crnn_gan"])


    def forward(self, x, pad_mask=None, embeddings=None):
        real_strong_preds, real_weak_preds = self.crnn(x, pad_mask, embeddings)

        B, T, F = real_strong_preds.shape
        real_embeddings = real_strong_preds.weak.reshape(B, -1)
        fake_spec = self.gan(real_embeddings)

        fake_strong_preds, fake_weak_preds = self.crnn(fake_spec)

        return real_strong_preds, real_weak_preds, fake_strong_preds, fake_weak_preds

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN_GAN, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
