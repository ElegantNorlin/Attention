from __future__ import division
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class Self_Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B*N*C
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B*C*N
        energy =  torch.bmm(proj_query,proj_key) # batch的matmul B*N*N
        attention = self.softmax(energy) # B * (N) * (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1, width*height) # B * C * N

        out = torch.bmm(proj_value,attention.permute(0,2,1) ) # B*C*N
        out = out.view(m_batchsize,C,width,height) # B*C*H*W

        out = self.gamma*out + x
        return out,attention


if __name__ == "__main__":
    self_attention = Self_Attention(3,None)
    self_attention.cuda()
    # bs,channels,height,width
    x = Variable(torch.rand([8, 3, 32, 32]).cuda())
    y = self_attention(x)

