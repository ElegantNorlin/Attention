from __future__ import division
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

'''
# 代码的注释是在input.shape = (8,64,32,32)的基础上进行注释的
query.view就是把长、宽像素拉直成一个维度，维度的维度值为原来的w * h
key.view就是把长、宽像素拉直成一个维度，维度的维度值为原来的w * h  
query和key正好是可以点乘的矩阵
经过softmax会给每一个像素分配一个权重值，每一行的权重之和为1
'''
class Self_Attention(nn.Module):
    # in_dim为输入、输出特征的通道维度数
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
        # proj_query.view.shape = torch.Size([8, 1024, 8])
        # self.query_conv(x).shape = torch.Size([8, 32, 32, 32])
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B*N*C
        # proj_key.view.shape = torch.Size([8, 8, 1024])
        # self.key_conv(x).shape = torch.Size([8, 32, 32, 32])
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B*C*N
        # energy.shape = torch.Size([8, 1024, 1024])
        energy =  torch.bmm(proj_query,proj_key) # batch的matmul B*N*N
        # attention.shape = torch.Size([8, 1024, 1024])
        # 过softmax之后会得到每一行像素的权重分配（每一行的权重之和为1）
        attention = self.softmax(energy) # B * (N) * (N)
        # proj_value.shape = torch.Size([8, 64, 1024])
        proj_value = self.value_conv(x).view(m_batchsize,-1, width*height) # B * C * N



        # out_bmm.shape = torch.Size([8, 64, 1024])
        out = torch.bmm(proj_value,attention.permute(0,2,1) ) # B*C*N
        # out_view.shape = torch.Size([8, 64, 32, 32])
        out = out.view(m_batchsize,C,width,height) # B*C*H*W


        # self.gamma.shape = torch.Size([1])，self.gamma为可训练参数，虽然数值全零，但在训练过程中会逐渐被赋值
        # self.gamma*out = torch.Size([8, 64, 32, 32])
        # out.shape = torch.Size([8, 64, 32, 32])
        out = self.gamma*out + x
        return out,attention


if __name__ == "__main__":
    self_attention = Self_Attention(64,None)
    self_attention.cuda()
    # bs,channels,height,width
    x = Variable(torch.rand([8, 64, 32, 32]).cuda())
    y = self_attention(x)

