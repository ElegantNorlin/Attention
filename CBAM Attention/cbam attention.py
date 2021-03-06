import numpy as np
import torch
from torch import nn
from torch.nn import init


# 这里类似于SE通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
    	# 这里定义的最大池化函数结果就是每一个通道的feature map只剩下一个像素（也就是该通道像素的最大值）
        max_result=self.maxpool(x)
        # 和最大值池化类似，这里剩下的一个像素是该通道像素的平均值
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

# 个人理解的空间注意力机制是对每个点像素的注意力权重分配
'''
在空间注意力机制中，先是在第二维度上做了最大池化和平均池化得到的其实就是最有代表性的
两种特征，他们的维度为(bs,1,h,w),然后将这两个tensor在dim=1第二维度相加。得到2通道的tensor
然后再做一次卷积，把通道数调整为1，也就是将两个tensor提取到的信息压缩到一起
然后我们得到一个单通道的tensor，然后过sigmoid激活函数，得到每一个像素点的权重。
'''
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    cbam = CBAMBlock(channel=512,reduction=16)
    output=cbam(input)
    print(output.shape)