import torch
from local_lib.mobile_darknet import MobileDarknet


class MainNet(torch.nn.Module):

    def __init__(self, cls_num, cent_num):
        super(MainNet, self).__init__()

        self.cent_num = cent_num
        self.features = torch.nn.Sequential(
            MobileDarknet(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(1024, cls_num, 1, 1, 0),
        )

    def forward(self, x):
        features = self.features(x)
        cls = self.classifier(features)

        centers = features.permute(0, 2, 3, 1)
        centers = centers.reshape(*centers.shape[:-1], self.cent_num, -1)
        centers = torch.mean(centers, dim=4)

        return centers.squeeze(), cls.squeeze()
