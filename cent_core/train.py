import os
import torch
import matplotlib.pyplot as plt

from local_lib.lookahead import Lookahead
from local_lib.miscs import ArgParse, Trainer
from cent_core.center_loss import CenterLoss


class Args(ArgParse):

    def __init__(self, description):
        super(Args, self).__init__(description=description)
        self.parser.add_argument("--beta", type=float, default=0., help="weights of center-loss")
        self.parser.add_argument("--alpha", type=float, default=5e-1, help="learning rate for center-loss layer")
        self.parser.add_argument("--scatter-dir", type=str, default="scatters", help="directory for feature-scatters pictures saving")


class Scatter:

    def __init__(self, cls_num):
        self.__cls_num = cls_num
        self.__colors = [f'C{i}' for i in range(cls_num)]
        self.reset()

    def reset(self):
        self.__features = []
        self.__labels = []

    def update(self, features, labels):
        self.__features.append(features.cpu().data)
        self.__labels.append(labels.cpu().data)

    def plot(self, _dir, prefix, _n):
        features = torch.cat(self.__features).numpy()
        labels = torch.cat(self.__labels).numpy()
        for label_idx in range(self.__cls_num):
            plt.scatter(features[labels == label_idx, 0], features[labels == label_idx, 1], c=self.__colors[label_idx], s=1)
        plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], loc='upper right')

        dirname = os.path.join(_dir, prefix)
        if not os.path.exists(dirname):
            os.makedirs(dirname, 0o775)
        save_name = os.path.join(dirname, 'epoch_' + str(_n) + '.png')
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()


class Train(Trainer):

    def __init__(self, train_dataset, val_dataset, model, cls_num, feat_num):
        self.cls_num = cls_num
        self.feat_num = feat_num

        super(Train, self).__init__(train_dataset, val_dataset, args=Args("CIFAR10 classifier trainer"))
        self._appendcell(['center_loss'])

        self.net = model.to(self.device)
        self.center_loss = CenterLoss(self.cls_num, self.feat_num).to(self.device)

        params = list(self.net.parameters()) + list(self.center_loss.parameters())
        self.optimizer = Lookahead(torch.optim.Adam(params, self.args.lr))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.value = 0.

        self.scatter = Scatter(cls_num)

    def train(self):
        self.net.train()
        self.scatter.reset()
        print(f"epochs: {self.epoch}")
        for i, (data, labels) in enumerate(self.train_loader, start=1):
            data, labels = data.to(self.device), labels.to(self.device)
            features, outputs = self.net(data)

            xent_loss = self.criterion(outputs, labels)
            cent_loss = self.center_loss(features, labels) * self.args.beta
            loss = xent_loss + cent_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.args.beta != 0:
                for param in self.center_loss.parameters():
                    param.grad.data *= self.args.alpha / (self.args.beta * self.args.lr)
            self.optimizer.step()

            self.scatter.update(features, labels)

            if i % self.args.print_freq == 0:
                print(f"[epoch: {self.epoch} - {i}/{len(self.train_loader)}]"
                      f"Loss: {loss.float()} - Loss_xent: {xent_loss.float()} - Loss_cent: {cent_loss.float()}")

        self.scatter.plot(self.args.scatter_dir, 'train', self.epoch)

    def validate(self):
        self.net.eval()
        self.scatter.reset()
        correct, valid, total = 0, 0, 0
        for data, labels in self.val_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            features, outputs = self.net(data)

            valid_mask = torch.softmax(outputs.data, dim=1) > 0.9
            predictions = torch.nonzero(valid_mask)
            total += labels.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels.data).sum().float()
            valid += (predictions[:, 1] == labels.data[predictions[:, 0]]).sum().float()

            self.scatter.update(features, labels)

        self.scatter.plot(self.args.scatter_dir, 'val', self.epoch)

        acc = 100 * correct / total
        val = 100 * valid / total
        print(f"[epochs: {self.epoch}]Valid: {val.float():.2f}% - Accuracy: {acc.float():.2f}%")
        return val.item()
