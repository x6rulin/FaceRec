import os
import sys
import torchvision

rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)

from cent_core.model import MainNet
from cent_core.train import Train

if __name__ == "__main__":
    data_dir = "data/cifar10"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, 0o775)

    _normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261), inplace=True)
    train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     _normalize,
                                                 ]))
    val_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=False,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   _normalize,
                                               ]))
    cls_num, feat_num = 10, 2

    net = MainNet(cls_num, feat_num, cfg=(1, 2, 4, 4, 2), drop=0.05, init_weights=True, bn=False, activation='selu', inplace=True)
    trainer = Train(train_dataset, val_dataset, net, cls_num, feat_num)
    trainer()
