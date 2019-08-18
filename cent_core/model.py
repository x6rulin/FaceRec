import torch
from local_lib.xception_darknet import XceptionDarknet


class MainNet(torch.nn.Module):

    def __init__(self, cls_num, feat_num):
        super(MainNet, self).__init__()

        self.features = torch.nn.Sequential(
            XceptionDarknet(),
        )
        self.scatters = torch.nn.Sequential(
            torch.nn.Conv2d(1024, feat_num, 1, 1, 0),
            torch.nn.BatchNorm2d(feat_num),
            torch.nn.PReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(feat_num, cls_num, 1, 1, 0),
        )

    def forward(self, x):
        features = self.features(x)
        scatters = self.scatters(features)
        cls = self.classifier(scatters)

        return scatters.squeeze(), cls.squeeze()
