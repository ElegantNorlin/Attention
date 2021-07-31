### SE Attention





![](https://github.com/ElegantAnkster/Attention/blob/main/SE%20Attention/Architecture%20Image/640.png?raw=true)

SENet是早期Attention，核心思想是学习 feature Channel 间的关系，以凸显feature Channel不同的重要度（也就是注意力分布），进而提高模型表现。

SE通道注意力机制:

* 平均池化，压缩空间，得到向量某种程度上具有全域性的感受野。相当于把每一个通道的像素取平均值，即每一个通道的feature map只有一个像素（也就是每个通道feature map的平均值）
* 通过一个 Sigmoid 的门获得 0~1 之间归一化的权重，经过特征参数打印，发现过sigmoid函数后，每一个通道都会获得一个0～1之间的权重数值，也就是我们需要的通道权重。