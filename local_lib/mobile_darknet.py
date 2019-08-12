"""self-defined Mobile-Darknet. """
import torch


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=True):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.PReLU(),
        )

    def forward(self, x):
        return self.sub_module(x)


class DownSampleLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels, 3, 2, 1, in_channels),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        mid_channels = in_channels * 2
        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, mid_channels, 1, 1, 0),
            ConvolutionalLayer(mid_channels, mid_channels, 3, 1, 1, mid_channels),
            ConvolutionalLayer(mid_channels, in_channels, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.sub_module(x)


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, _n):
        super(ResidualBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            *[ResidualLayer(in_channels) for _ in range(_n)],
        )

    def forward(self, x):
        return self.sub_module(x)


class MobileDarknet(torch.nn.Module):

    def __init__(self):
        super(MobileDarknet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),

            ResidualBlock(64, 1),
            DownSampleLayer(64, 128),

            ResidualBlock(128, 2),
            DownSampleLayer(128, 256),

            ResidualBlock(256, 8),
            DownSampleLayer(256, 512),

            ResidualBlock(512, 8),
            DownSampleLayer(512, 1024),

            ResidualBlock(1024, 4)
        )

    def forward(self, x):
        return self.sub_module(x)
