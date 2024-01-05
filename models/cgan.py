import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x 12 x 64 x 6
            self._block(channels_img+1, features_d, 1, 1, 0),
            self._block(features_d, features_d*2, 1, 1, 0),
            self._block(features_d*2, features_d*4, 1, 1, 0),
            self._block(features_d*4, 1, 1, 1, 0),
            nn.Flatten(start_dim=1),
            nn.Linear(12*64*6, 1),
            nn.Sigmoid(),
        )
        self.embed = nn.Embedding(num_classes, 12*64*6)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        # print(x.shape)
        embedding= self.embed(labels.squeeze(1)).view(x.shape[0],1, 12, 64, 6)
        # print(embedding.shape)
        x = torch.cat([x, embedding], dim=1)
        # print(x.shape)
        x = self.disc(x.double())
        return x



class Generator(nn.Module):
    def __init__(
            self, channels_noise, channels_img, features_g, num_classes,img_size,embed_size):
        super(Generator, self).__init__()
        self.img_size=img_size
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise + num_classes, 256, 4, 1, 0),  # img: 4x4
            self._block(256, 128, 2, 2, 0),  # img: 8x8v
            self._block( 128, 64, 1, 1, 0),  # img: 8x8v
            self._block(64, 32, 1, 1, 0),  # img: 8x8v
            self._block( 32, 9, 1, 1, 0),  # img: 8x8v
            nn.Tanh(),
        )
        self.embed= nn.Embedding(num_classes,num_classes)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # p = self.net(x)
        # print(self.net(x).shape)
        embedding = self.embed(labels.squeeze(1))
        x = x.view(x.shape[0],-1)
        # print(embedding.shape)
        # print(x.shape)
        x = torch.cat([x, embedding], dim=-1)
        x = x.view(x.shape[0], -1, 1,1, 1)
        return self.net(x).reshape(-1, 1, 12, 64, 6)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)