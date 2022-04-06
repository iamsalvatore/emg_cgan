import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x 12 x 64 x 6
            self._block(channels_img, features_d, 1, 1, 0),
            self._block(features_d, features_d*2, 1, 1, 0),
            self._block(features_d*2, features_d*4, 1, 1, 0),
            self._block(features_d*4, 1, 1, 1, 0),
            nn.Flatten(start_dim=1),
            nn.Linear(12*64*6, 1),
            nn.Sigmoid(),
            # self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            # nn.Conv3d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.disc(x.double())
        return x



class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, 256, 4, 1, 0),  # img: 4x4
            self._block(256, 128, 2, 2, 0),  # img: 8x8v
            self._block( 128, 64, 1, 1, 0),  # img: 8x8v
            self._block(64, 32, 1, 1, 0),  # img: 8x8v
            self._block( 32, 9, 1, 1, 0),  # img: 8x8v
            # self._block(128, 64, 2, 2, 0),
            # self._block(64, 32, 2, 2, 0),
            # self._block(32, 12, 2, 2, 0),
            # self._block(8, 4, 1, 1, 0),
            # self._block(4, 2, 1, 1, 0),
            # self._block(features_g * 8, features_g * 4, 2, 2, 0),  # img: 16x16
            # self._block(features_g * 4, features_g * 2, 2, 2, 0),  # img: 32x32
            # nn.ConvTranspose3d(
            #     features_g * 2, channels_img, kernel_size=2, stride=1, padding=0
            # ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # p = self.net(x)
        print(self.net(x).shape)
        return self.net(x).reshape(-1, 1, 12, 64, 6)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)





# class Discriminator2(nn.Module):
#     def __init__(self, channels_img, features_d):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             # input: N x channels_img x 64 x 64
#             nn.Conv2d(channels_img, features_d, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(0.2),
#             # _block(in_channels, out_channels, kernel_size, stride, padding)
#             # self._block(features_d, 1, 1, 1, 0),
#             # self._block(features_d * 2, features_d * 4, 4, 2, 1),
#             # self._block(features_d * 4, features_d * 8, 4, 2, 1),
#             # After all _block img output is 4x4 (Conv2d below makes into 1x1)
#             # nn.Conv3d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
#         )
#
#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.Conv3d(
#                 in_channels, out_channels, kernel_size, stride, padding, bias=False,
#             ),
#             nn.InstanceNorm3d(out_channels, affine=True),
#             nn.LeakyReLU(0.2),
#         )

    # def forward(self, x):
    #     x = self.disc(x.double())
    #     return self.disc(x)