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

        self.branches = torch.nn.ModuleList(
            [self._branch(in_channels, out_channels, kernel_size)
             for out_channels, kernel_size in zip(channels, kernels_size)]
        )
        self.merge = torch.nn.Sequential(
            ConvolutionalLayer(sum(channels), sum(channels), 1, 1, 0),
            torch.nn.Dropout2d(inplace=True),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]

        return self.merge(torch.cat(outputs, dim=1))

    @staticmethod
    def _branch(in_channels, out_channels, kernel_size):
        bottle = [ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)]
        for _ in range(1, kernel_size, 2):
            bottle.append(ConvolutionalLayer(out_channels, out_channels, 3, 1, 1, out_channels))

        return torch.nn.Sequential(*bottle)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels, kernels_size):
        super(ResidualLayer, self).__init__()

        channels = self._split(in_channels, len(kernels_size))
        self.sub_module = MDConv2dLayer(in_channels, channels, kernels_size)

    def forward(self, x):
        return x + self.sub_module(x)

    @staticmethod
    def _split(channels, _n):
        slices = []
        for _ in range(_n - 1):
            if channels < 2: break
            slices.append(channels - (channels >> 1))
            channels >>= 1
        slices.append(channels)

        return slices


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, kernels_size, _n):
        super(ResidualBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            *[ResidualLayer(in_channels, kernels_size) for _ in range(_n)],
        )

    def forward(self, x):
        return self.sub_module(x)


class XceptionDarknet(torch.nn.Module):

    def __init__(self, cfg):
        super(XceptionDarknet, self).__init__()

        channels = 32
        layers = [torch.nn.Sequential(
            ConvolutionalLayer(3, channels, 1, 1, 0),
            ConvolutionalLayer(channels, channels, 3, 1, 1, channels),
            ConvolutionalLayer(channels, channels, 1, 1, 0),
        )]
        for _n in cfg:
            tmp, channels = channels, 2 * channels
            layers.append(torch.nn.Sequential(
                DownSampleLayer(tmp, channels),
                ResidualBlock(channels, (3, 5, 7, 9), _n)
            ))

        self.sub_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.sub_module(x)
