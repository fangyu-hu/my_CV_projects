from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class DecodeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, out_padding=1):
        super(DecodeConvBlock, self).__init__()
        self.de_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, output_padding=out_padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x, is_act=True):
        x = self.de_conv(x)
        if is_act:
            x = torch.relu(self.bn(x))
        return x


class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=2)
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels)
        self.conv3 = ConvBlock(in_channels=out_channels, out_channels=out_channels)
        self.conv4 = ConvBlock(in_channels=out_channels, out_channels=out_channels)

        self.short_cut = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)

        short_cut = self.short_cut(x)

        out2 = self.conv3(out1 + short_cut)
        out2 = self.conv4(out2)

        return out1 + out2


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodeBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, padding=0)
        self.de_conv = DecodeConvBlock(in_channels=in_channels // 4, out_channels=in_channels // 4)
        self.conv3 = ConvBlock(in_channels=in_channels // 4, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.de_conv(x)
        x = self.conv3(x)
        return x


class LinkNet(nn.Module):
    def __init__(self):
        super(LinkNet, self).__init__()
        self.init_conv = ConvBlock(in_channels=3, out_channels=64, stride=2, kernel_size=7, padding=3)
        self.init_maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.encode_1 = EncodeBlock(in_channels=64, out_channels=64)
        self.encode_2 = EncodeBlock(in_channels=64, out_channels=128)
        self.encode_3 = EncodeBlock(in_channels=128, out_channels=256)
        self.encode_4 = EncodeBlock(in_channels=256, out_channels=512)

        self.decode_4 = DecodeBlock(in_channels=512, out_channels=256)
        self.decode_3 = DecodeBlock(in_channels=256, out_channels=128)
        self.decode_2 = DecodeBlock(in_channels=128, out_channels=64)
        self.decode_1 = DecodeBlock(in_channels=64, out_channels=64)

        self.deconv_out1 = DecodeConvBlock(in_channels=64, out_channels=32)
        self.conv_out = ConvBlock(in_channels=32, out_channels=32)
        self.deconv_out2 = DecodeConvBlock(in_channels=32, out_channels=2, kernel_size=2, padding=0, out_padding=0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_maxpool(x)

        e1 = self.encode_1(x)
        e2 = self.encode_2(e1)
        e3 = self.encode_3(e2)
        e4 = self.encode_4(e3)

        d4 = self.decode_4(e4)
        d3 = self.decode_3(d4 + e3)
        d2 = self.decode_2(d3 + e2)
        d1 = self.decode_1(d2 + e1)

        f1 = self.deconv_out1(d1)
        f2 = self.conv_out(f1)
        f3 = self.deconv_out2(f2)
        return f3

