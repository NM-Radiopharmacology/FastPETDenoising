"""
U-NET 3D IMPLEMENTATION
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
        self.activ = nn.ReLU()

        self.conv1.weight = nn.init.kaiming_normal_(self.conv1.weight)
        self.conv1.bias = nn.init.zeros_(self.conv1.bias)
        self.conv2.weight = nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2.bias = nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pooling):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        if pooling == 'average':
            self.pool = nn.AvgPool3d((2, 2, 2))
        else:
            self.pool = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, inputs):

        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)

        self.up.weight = nn.init.zeros_(self.up.weight)
        self.up.bias = nn.init.zeros_(self.up.bias)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat((x, skip), 1)
        x = self.conv(x)

        return x


class UNet3D(nn.Module):

    def __init__(self, n_channels=None, pooling=None):

        if n_channels is None:
            n_channels = 64

        if pooling is None:
            pooling = 'average'

        self.pooling = pooling

        super().__init__()

        """ Encoder """
        self.e1 = EncoderBlock(1, n_channels, pooling=pooling)
        self.e2 = EncoderBlock(n_channels, n_channels * 2, pooling=pooling)
        self.e3 = EncoderBlock(n_channels * 2, n_channels * 4, pooling=pooling)

        """ Bottleneck """
        self.b = ConvBlock(n_channels * 4, n_channels * 8)

        """ Decoder """
        self.d1 = DecoderBlock(n_channels * 8, n_channels * 4)
        self.d2 = DecoderBlock(n_channels * 4, n_channels * 2)
        self.d3 = DecoderBlock(n_channels * 2, n_channels)

        """ Classifier """
        self.outputs = nn.Conv3d(n_channels, 1, kernel_size=1, padding=0)
        self.final_activ = nn.ReLU()

    def forward(self, inputs):

        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        """ Bottleneck """
        b = self.b(p3)

        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        """ Classifier """
        outputs0 = self.outputs(d3)
        outputs = self.final_activ(outputs0)

        return outputs
