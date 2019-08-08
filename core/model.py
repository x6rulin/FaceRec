import torch
from core.mobile_darknet import MobileDarknet


class MainNet(torch.nn.Module):

    def __init__(self, cls_num, feat_num):
        super(MainNet, self).__init__()

        self.features = torch.nn.Sequential(
            MobileDarknet(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.centers = torch.nn.Sequential(
            torch.nn.Conv2d(1024, feat_num, 1, 1, 0),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(1024, cls_num, 1, 1, 0),
        )

    def forward(self, x):
        features = self.features(x)
        centors = self.centers(features)
        cls = self.classifier(features)

        return centors.squeeze(), cls.squeeze()
