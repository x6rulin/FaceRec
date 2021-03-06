import torch


class CenterLoss(torch.nn.Module):

    def __init__(self, cls_num, feat_num):
        super(CenterLoss, self).__init__()

        self.cls_num = cls_num
        self.center = torch.nn.Parameter(torch.randn(cls_num, feat_num))

    def forward(self, xs, ys):
        center_exp = self.center.index_select(dim=0, index=ys.long())
        count = torch.histc(ys.float(), bins=self.cls_num, min=0, max=self.cls_num - 1)
        count_exp = count.index_select(dim=0, index=ys.long()).float()
        loss = torch.sum(torch.mean(torch.pow(xs - center_exp, 2), dim=1) / (2. * count_exp)) / self.cls_num
        return loss
