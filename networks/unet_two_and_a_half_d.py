"""
2.5D U-NET IMPLEMENTATION
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.in_channels = in_channels

        if in_channels == 3:
            self.conv0 = nn.Conv2d(in_channels, out_channels=1, kernel_size=3, padding='same', padding_mode='reflect')

            self.conv1 = nn.Conv2d(1, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
            self.relu = nn.ReLU()

            self.conv0.weight = nn.init.kaiming_normal_(self.conv0.weight)
            self.conv0.bias = nn.init.zeros_(self.conv0.bias)
            self.conv1.weight = nn.init.kaiming_normal_(self.conv1.weight)
            self.conv1.bias = nn.init.zeros_(self.conv1.bias)
            self.conv2.weight = nn.init.kaiming_normal_(self.conv2.weight)
            self.conv2.bias = nn.init.zeros_(self.conv2.bias)

        else:

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
            self.relu = nn.ReLU()

            self.conv1.weight = nn.init.kaiming_normal_(self.conv1.weight)
            self.conv1.bias = nn.init.zeros_(self.conv1.bias)
            self.conv2.weight = nn.init.kaiming_normal_(self.conv2.weight)
            self.conv2.bias = nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs):

        if self.in_channels == 3:
            x = self.conv0(inputs)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
        else:
            x = self.conv1(inputs)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d((2, 2))

    def forward(self, inputs):

        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)

        # with torch.no_grad(): IF NECESSARY
        self.up.weight = nn.init.zeros_(self.up.weight)
        self.up.bias = nn.init.zeros_(self.up.bias)

    def forward(self, inputs, skip):

        x = self.up(inputs)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)

        return x


class UNet25D(nn.Module):

    def __init__(self, in_channels=None):

        super().__init__()

        if in_channels is None:
            in_channels = 1
        else:
            in_channels = in_channels

        self.in_channels = in_channels

        """ Encoder """
        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.b = ConvBlock(512, 1024)

        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.final_activ = nn.ReLU()

    def forward(self, inputs):

        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs0 = self.outputs(d4)
        outputs = self.final_activ(outputs0)

        return outputs
