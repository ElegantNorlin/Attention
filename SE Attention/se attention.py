from __future__ import division
import torch
import torch.nn as nn
import os
from torch.autograd import Variable

'''
代码的注视都是基于输入特征为input.shape = (8,64,32,32)
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 压缩空间
        self.fc = nn.Sequential(
        	# torch.Size([8, 4])
            nn.Linear(channel, channel // reduction, bias=False),
            # torch.Size([8, 4])
            nn.ReLU(inplace=True),
            # torch.Size([8, 64])
            nn.Linear(channel // reduction, channel, bias=False),
            # torch.Size([8, 64])
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # avg_pool(x).shape = torch.Size([8, 64, 1, 1])
        # avg_pool(x).view(b, c).shape = torch.Size([8, 64])
        y = self.avg_pool(x).view(b, c)
        # torch.Size([8, 64, 1, 1])
        y = self.fc(y).view(b, c, 1, 1)
        # y.expand_as(x).shape = torch.Size([8, 64, 32, 32])
        # x * y.expand_as(x).shape = torch.Size([8, 64, 32, 32])
        return x * y.expand_as(x)


if __name__ == "__main__":
    se = SELayer(16)
    se.cuda()
    x = Variable(torch.rand([8, 16, 32, 32]).cuda())
    y = se(x)