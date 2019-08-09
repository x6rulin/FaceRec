import os
import torch
import matplotlib.pyplot as plt

from local_lib.lookahead import Lookahead
from local_lib.miscs import ArgParse, Trainer
from cent_core.center_loss import CenterLoss


class Args(ArgParse):

    def __init__(self):
        super(Args, self).__init__()
        self.parser.add_argument("--beta", type=float, default=0., help="weights of center-loss")
        self.parser.add_argument("--alpha", type=float, default=0., help="learning rate for center-loss layer")
        self.parser.add_argument("--scatter-dir", type=str, default="scatters", help="directory for feature-scatters pictures saving")


class Train(Trainer):

    def __init__(self, train_dataset, val_dataset, cls_num, feat_num, model):
        self.cls_num = cls_num
        self.feat_num = feat_num
        super(Train, self).__init__(train_dataset, val_dataset, model, args=Args())
        self.center_loss = CenterLoss(self.cls_num, self.feat_num).to(self.device)

        params = list(self.net.parameters()) + list(self.center_loss.parameters())
        self.optimizer = Lookahead(torch.optim.Adam(params, self.args.lr))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.value = 0.

    def train(self):
        self.net.train()
        print(f"epochs: {self.epoch}")
        all_features, all_labels = [], []
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

            all_features.append(features.cpu().data)
            all_labels.append(labels.cpu().data)

            if i % self.args.print_freq == 0:
                print(f"[epoch: {self.epoch} - {i}/{len(self.train_loader)}]"
                      f"Loss: {loss.float()} - Loss_xent: {xent_loss.float()} - Loss_cent: {cent_loss.float()}")

        all_features = torch.cat(all_features).numpy()
        all_labels = torch.cat(all_labels).numpy()
        self.plot_features(all_features, all_labels, prefix='train')

    def validate(self):
        self.net.eval()
        all_features, all_labels = [], []
        correct, total = 0, 0
        for data, labels in self.val_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            features, outputs = self.net(data)

            valid_mask = torch.softmax(outputs.data, dim=1) > 0.9
            predictions = torch.nonzero(valid_mask)
            total += labels.size(0)
            correct += (predictions[:, 1] == labels.data[predictions[:, 0]]).sum().float()

            all_features.append(features.cpu().data)
            all_labels.append(labels.cpu().data)

        all_features = torch.cat(all_features).numpy()
        all_labels = torch.cat(all_labels).numpy()
        self.plot_features(all_features, all_labels, prefix='val')

        acc = correct * 100. / total.float()
        print(f"[epochs: {self.epoch}]Accuracy: {acc.float():.2f}%")
        return acc.item()

    def plot_features(self, features, labels, prefix):
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for label_idx in range(self.cls_num):
            plt.scatter(
                features[labels == label_idx, 0],
                features[labels == label_idx, 1],
                c=colors[label_idx],
                s=1,
            )
        plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], loc='upper right')
        dirname = os.path.join(self.args.scatter_dir, prefix)
        if not os.path.exists(dirname):
            os.makedirs(dirname, 0o775)
        save_name = os.path.join(dirname, 'epoch_' + str(self.epoch) + '.png')
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
