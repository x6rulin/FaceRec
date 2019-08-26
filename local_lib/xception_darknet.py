"""self-defined Xception-Darknet. """
import torch


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False, bn=True, selu=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            *[torch.nn.BatchNorm2d(out_channels)][:bn],
            [torch.nn.PReLU(out_channels), torch.nn.SELU(inplace=True)][selu],
        )

    def forward(self, x):
        return self.sub_module(x)


class DownSampleLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(DownSampleLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels, 3, 2, 1, in_channels, **kwargs),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0, **kwargs),
        )

    def forward(self, x):
        return self.sub_module(x)


class MDConv2dLayer(torch.nn.Module):

    def __init__(self, in_channels, channels, kernels_size, **kwargs):
        super(MDConv2dLayer, self).__init__()

        self.branches = torch.nn.ModuleList(
            [self._branch(in_channels, out_channels, kernel_size, **kwargs)
             for out_channels, kernel_size in zip(channels, kernels_size)]
        )
        self.merge = torch.nn.Sequential(
            ConvolutionalLayer(sum(channels), sum(channels), 1, 1, 0, **kwargs),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]

        return self.merge(torch.cat(outputs, dim=1))

    @staticmethod
    def _branch(in_channels, out_channels, kernel_size, **kwargs):
        bottle = [ConvolutionalLayer(in_channels, out_channels, 1, 1, 0, **kwargs)]
        for _ in range(1, kernel_size, 2):
            bottle.append(ConvolutionalLayer(out_channels, out_channels, 3, 1, 1, out_channels, **kwargs))

        return torch.nn.Sequential(*bottle)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels, kernels_size, **kwargs):
        super(ResidualLayer, self).__init__()

        channels = self._split(in_channels, len(kernels_size))
        self.sub_module = MDConv2dLayer(in_channels, channels, kernels_size, **kwargs)

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

    def __init__(self, in_channels, kernels_size, _n, **kwargs):
        super(ResidualBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            *[ResidualLayer(in_channels, kernels_size, **kwargs) for _ in range(_n)],
        )

    def forward(self, x):
        return self.sub_module(x)


class XceptionDarknet(torch.nn.Module):

    def __init__(self, cfg, **kwargs):
        super(XceptionDarknet, self).__init__()

        channels = 32
        layers = [torch.nn.Sequential(
            ConvolutionalLayer(3, channels, 1, 1, 0, **kwargs),
            ConvolutionalLayer(channels, channels, 3, 1, 1, channels, **kwargs),
            ConvolutionalLayer(channels, channels, 1, 1, 0, **kwargs),
        )]
        for _n in cfg:
            tmp, channels = channels, 2 * channels
            layers.append(torch.nn.Sequential(
                DownSampleLayer(tmp, channels, **kwargs),
                ResidualBlock(channels, (3, 5, 7, 9), _n, **kwargs),
            ))

        self.sub_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.sub_module(x)
