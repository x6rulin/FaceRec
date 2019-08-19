"""self-defined Xception-Darknet. """
import torch


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
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


class MDConv2dLayer(torch.nn.Module):

    def __init__(self, in_channels, channels, kernels_size):
        super(MDConv2dLayer, self).__init__()

        self.branches = [self._branch(in_channels, out_channels, kernel_size)
                         for out_channels, kernel_size in zip(channels, kernels_size)]

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]

        return torch.cat(outputs, dim=1)

    @staticmethod
    def _branch(in_channels, out_channels, kernel_size):
        bottle = [ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)]
        for _ in range(1, kernel_size, 2):
            bottle.append(torch.nn.Sequential(
                ConvolutionalLayer(out_channels, out_channels, 3, 1, 1, out_channels),
                ConvolutionalLayer(out_channels, out_channels, 1, 1, 0),
            ))

        return torch.nn.Sequential(*bottle)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels, kernels_size):
        super(ResidualLayer, self).__init__()

        channels = self._split(in_channels, len(kernels_size))
        self.sub_module = MDConv2dLayer(in_channels, channels, kernels_size)

    def forward(self, x):
        return x + self.sub_module(x)

    @staticmethod
    def _split(in_channels, _n):
        slices = []
        for _ in range(_n - 1):
            tmp = in_channels // 2
            if tmp == 0: break
            slices.append(tmp)
            in_channels -= tmp
        slices.append(in_channels)

        return slices[-1:] + slices[:-1]


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, kernels_size, _n):
        super(ResidualBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            *[ResidualLayer(in_channels, kernels_size) for _ in range(_n)],
        )

    def forward(self, x):
        return self.sub_module(x)


class XceptionDarknet(torch.nn.Module):

    def __init__(self):
        super(XceptionDarknet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),

            ResidualBlock(64, (1, 3, 5, 7, 9), 1),
            DownSampleLayer(64, 128),

            ResidualBlock(128, (1, 3, 5, 7, 9), 2),
            DownSampleLayer(128, 256),

            ResidualBlock(256, (1, 3, 5, 7, 9), 8),
            DownSampleLayer(256, 512),

            ResidualBlock(512, (1, 3, 5, 7, 9), 8),
            DownSampleLayer(512, 1024),

            ResidualBlock(1024, (1, 3, 5, 7, 9), 4)
        )

    def forward(self, x):
        return self.sub_module(x)
