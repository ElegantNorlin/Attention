## CBAM的一些理解

### spatial attention

* 在空间注意力机制中，先是在每一通道的通同一个位置上做了最大值和平均值，得到的特征其实就是最有代表性的两种特征，他们的维度皆为(bs,1,h,w)。
* 然后将这两个tensor在dim=1第二维度相加。得到2通道的tensor。
* 再做一次卷积，把通道数调整为1，也就是将两个tensor提取到的信息压缩到一起，
  我们得到一个单通道的tensor。
* 最后过sigmoid激活函数，得到每一个像素点的权重。

通俗易懂的理解：

空间注意力机制就是给每一个像素点分配对应的权值，然后把输入与注意力后得到的权值相乘，得到注意力后的tensor

