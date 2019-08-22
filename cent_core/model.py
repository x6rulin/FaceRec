import torch
from local_lib.xception_darknet import XceptionDarknet


class MainNet(torch.nn.Module):

    def __init__(self, cls_num, feat_num, cfg=(1, 2, 8, 8, 4), drop=0.1, init_weights=True):
        super(MainNet, self).__init__()

        self.features = torch.nn.Sequential(
            XceptionDarknet(cfg, drop),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.scatters = torch.nn.Sequential(
            torch.nn.Linear(1024 * 7 * 7, 4096),
            torch.nn.SELU(inplace=True),
            torch.nn.Linear(4096, feat_num),
            torch.nn.SELU(inplace=True),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(feat_num, cls_num),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        scatters = self.scatters(features.flatten(1))
        cls = self.classifier(scatters)

        return scatters.squeeze(), cls.squeeze()

    def _initialize_weights(self):
        for _m in self.modules():
            if isinstance(_m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(_m.weight, mode='fan_in', nonlinearity='conv2d')
                if _m.bias is not None:
                    torch.nn.init.constant_(_m.bias, 0)
            elif isinstance(_m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(_m.weight, 1)
                torch.nn.init.constant_(_m.bias, 0)
            elif isinstance(_m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(_m.weight, mode='fan_in', nonlinearity='linear')
                torch.nn.init.constant_(_m.bias, 0)
