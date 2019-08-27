import torch
from local_lib.xception_darknet import XceptionDarknet, _activate


class MainNet(torch.nn.Module):

    def __init__(self, cls_num, feat_num, cfg=(1, 2, 8, 8, 4), drop=0.25, init_weights=False, bn=True, activation='relu', **kwargs):
        super(MainNet, self).__init__()

        self.features = torch.nn.Sequential(
            XceptionDarknet(cfg, bn=bn, activation=activation, **kwargs),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        _act_layers = []
        if drop > 0:
            _act_layers.append([torch.nn.Dropout(drop, inplace=True), torch.nn.AlphaDropout(drop, inplace=True)][activation == 'selu'])
        _act_layers.extend(_activate(activation, **kwargs))
        self.scatters = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, 1, 1, 0),
            *_act_layers,
            torch.nn.Conv2d(512, feat_num, 1, 1, 0),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(feat_num, cls_num, 1, 1, 0),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        scatters = self.scatters(features)
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
