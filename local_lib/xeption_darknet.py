"""self-defined Xeption-Darknet. """
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


class XeptionLayer(torch.nn.Module):

    def __init__(self, in_channels, c1, c3, c5, c7, c9):
        super(XeptionLayer, self).__init__()

        self.branch_1 = ConvolutionalLayer(in_channels, c1, 1, 1, 0)
        self.branch_3 = self._branch(in_channels, c3, 3)
        self.branch_5 = self._branch(in_channels, c5, 5)
        self.branch_7 = self._branch(in_channels, c7, 7)
        self.branch_9 = self._branch(in_channels, c9, 9)

    def forward(self, x):
        branch1x1 = self.branch_1(x)
        branch3x3 = self.branch_3(x)
        branch5x5 = self.branch_5(x)
        branch7x7 = self.branch_7(x)
        branch9x9 = self.branch_9(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch7x7, branch9x9]
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _branch(in_channels, out_channels, kernel_size):
        mid_channels = out_channels // 2
        bottle = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, mid_channels, 1, 1, 0),
            ConvolutionalLayer(mid_channels, mid_channels, kernel_size, 1, kernel_size // 2, mid_channels),
            ConvolutionalLayer(mid_channels, out_channels, 1, 1, 0),
        )

        return bottle


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        c1, c3, c5, c7, c9 = self._split(in_channels, 5)
        self.sub_module = XeptionLayer(in_channels, c1, c3, c5, c7, c9)

    def forward(self, x):
        return x + self.sub_module(x)

    @staticmethod
    def _split(in_channels, _n):
        slices = []
        for _ in range(_n - 1):
            tmp = in_channels // 3
            slices.append(tmp)
            in_channels -= tmp
        slices.append(in_channels)

        return slices[-1:] + slices[:-1]


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, _n):
        super(ResidualBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            *[ResidualLayer(in_channels) for _ in range(_n)],
        )

    def forward(self, x):
        return self.sub_module(x)


class XeptionDarknet(torch.nn.Module):

    def __init__(self):
        super(XeptionDarknet, self).__init__()

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
