import torch
import torch.nn as nn
from torch.autograd.function import Function
import numpy as np

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        loss = self.centerlossfunc(feat, label, self.centers)
        loss /= (batch_size if self.size_average else 1)
        return loss

class CenterDistanceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True,bia = 0.4):
        super(CenterDistanceLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossDisFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.num_classes = num_classes
        self.bia = bia

    def compute_center_distance(self, centers):
        # print('compute_center_distance center size = ',center.size())
        shuffle_i = np.random.permutation(self.num_classes)
        # print('shuffle_i                   = ', shuffle_i[:10]," ,  end = ",shuffle_i[-10:])
        cha = centers[shuffle_i[:self.num_classes // 2]] - centers[shuffle_i[self.num_classes // 2:]]
        # print('compute_center_distance center_o size = ', center_o.size())
        # cha = self.centers[:self.num_classes//2, :self.feat_dim // 2] - self.centers[self.num_classes//2:, :self.feat_dim // 2]
        # print('origin                  cha = ', cha.size())
        distance = torch.pow(cha, 2)
        # print('origin                  distance = ', distance.data[16])
        s = torch.sum(distance, 1)
        # print('sum                  distance = ', s.data[15:20])
        sq = torch.sqrt(s)
        # print('sqrt                  distance = ', sq.data[15:20])
        me = torch.mean(sq)
        # print('mean                  distance = ', me.data)

        return me

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))

        loss = self.centerlossfunc(feat, label, self.centers)
        loss /= (batch_size if self.size_average else 1)

        distance = self.compute_center_distance(self.centers[:, :self.feat_dim // 4])
        # print('compute_center_distance distance = ', distance.data)
        # d_loss = torch.abs(torch.log10(distance + 0.5))
        d_loss = torch.reciprocal(distance + self.bia)
        return loss,d_loss,distance


class CenterlossDisFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        b, c = feature.size()
        weight = torch.cat([torch.ones((b, (c // 16) * 2)) / float(100), torch.ones((b, (c // 16) * 14))], dim=1).cuda()
        centers_batch = centers.index_select(0, label.long())

        return ((feature - centers_batch).pow(2) * weight ).sum() / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        b,c = feature.size()

        diff = centers_batch - feature
        weight = torch.cat([torch.ones((b,(c//16)*2))/float(100),torch.ones((b,(c//16)*14))],dim=1).cuda()
        # print("center loss weight 0 = ",weight[0,0]," 256 = ",weight[0,256])
        # init every iteration
        counts = centers.new(centers.size(0)).fill_(1)
        ones = centers.new(label.size(0)).fill_(1)
        grad_centers = centers.new(centers.size()).fill_(0)

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff * weight, None, grad_centers

class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new(centers.size(0)).fill_(1)
        ones = centers.new(label.size(0)).fill_(1)
        grad_centers = centers.new(centers.size()).fill_(0)

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff, None, grad_centers


def main(test_cuda=False):
    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")
    ct = CenterLoss(10,2).to(device)
    y = torch.Tensor([0,0,2,1]).to(device)
    feat = torch.zeros(4,2).to(device).requires_grad_()
    print (list(ct.parameters()))
    print (ct.centers.grad)
    out = ct(y,feat)
    print(out.item())
    out.backward()
    print(ct.centers.grad)
    print(feat.grad)

if __name__ == '__main__':
    torch.manual_seed(999)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=True)
