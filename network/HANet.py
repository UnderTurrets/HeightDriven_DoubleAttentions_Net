'''
Created by Han Xu
email:736946693@qq.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.PosEmbedding import PosEmbedding1D, PosEncoding1D  # 自定义的位置嵌入模块

class HANet_Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                 pos_rfactor=8, pooling='mean', dropout_prob=0.0, pos_noise=0.0):
        super(HANet_Conv, self).__init__()

        # 参数初始化
        self.pooling = pooling    # 池化方式，可选mean或max
        self.pos_injection = pos_injection  # 位置注入方式，1表示注入在输入特征层，2表示注入在第一次卷积后的输出
        self.layer = layer    # 卷积层数，可选2或3
        self.dropout_prob = dropout_prob  # dropout概率
        self.sigmoid = nn.Sigmoid()    # sigmoid函数

        if r_factor > 0:
            mid_1_channel = math.ceil(in_channel / r_factor)
            # 第一次卷积输出通道数，向上取整
        elif r_factor < 0:
            r_factor = r_factor * -1
            mid_1_channel = in_channel * r_factor    # 最终通道数，负值时为扩大倍数

        # Dropout
        if self.dropout_prob > 0:
            self.dropout = nn.Dropout2d(self.dropout_prob)

        # 池化层
        if self.pooling == 'mean':
            self.rowpool = nn.AdaptiveAvgPool2d((128 // pos_rfactor, 1))
        else:
            self.rowpool = nn.AdaptiveMaxPool2d((128 // pos_rfactor, 1))  # Adaptive pool方法，自适应池化

        # 第一个通道的注意力机制
        self.attention_first = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=mid_1_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),    # 1D卷积操作
            nn.BatchNorm1d(mid_1_channel),    # BN层
            nn.ReLU(inplace=True),    # ReLU激活函数
        )

        # 第二个通道的注意力机制
        if layer == 2:
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))    # 1D卷积操作，卷积核大小为kernel_size
        elif layer == 3:
            mid_2_channel = (mid_1_channel * 2)    # 第二个通道输出通道数
            self.attention_second = nn.Sequential(
                nn.Conv1d(in_channels=mid_1_channel, out_channels=mid_2_channel,
                          kernel_size=3, stride=1, padding=1, bias=True),   # 1D卷积操作，卷积核大小为3
                nn.BatchNorm1d(mid_2_channel),    # BN层
                nn.ReLU(inplace=True),
            )
            self.attention_third = nn.Sequential(
                nn.Conv1d(in_channels=mid_2_channel, out_channels=out_channel,
                          kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True))    # 1D卷积操作，卷积核大小为kernel_size



        # 位置嵌入模块
        if pos_rfactor > 0:
            if is_encoding == 0:
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEmbedding1D(pos_rfactor, dim=in_channel, pos_noise=pos_noise)
                    # 在输入特征层进行位置嵌入
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEmbedding1D(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
                    # 在第一次卷积后输出进行位置嵌入
            elif is_encoding == 1:
                if self.pos_injection == 1:
                    self.pos_emb1d_1st = PosEncoding1D(pos_rfactor, dim=in_channel, pos_noise=pos_noise)
                    # 在输入特征层进行位置编码
                elif self.pos_injection == 2:
                    self.pos_emb1d_2nd = PosEncoding1D(pos_rfactor, dim=mid_1_channel, pos_noise=pos_noise)
                    # 在第一次卷积后输出进行位置编码
            else:
                print("Not supported position encoding")
                exit()

    def forward(self, x, out, pos=None, return_attention=False, return_posmap=False, attention_loss=False):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        # HANet-ConV的前向传播过程

        H = out.size(2)    # Height

        x1d = self.rowpool(x).squeeze(3)    # 池化

        # 位置嵌入或编码
        if pos is not None and self.pos_injection == 1:
            if return_posmap:
                x1d, pos_map1 = self.pos_emb1d_1st(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_1st(x1d, pos)

        # Dropout
        if self.dropout_prob > 0:
            x1d = self.dropout(x1d)
        x1d = self.attention_first(x1d)    # 第一个通道的注意力机制

        # 位置嵌入或编码
        if pos is not None and self.pos_injection == 2:
            if return_posmap:
                x1d, pos_map2 = self.pos_emb1d_2nd(x1d, pos, True)
            else:
                x1d = self.pos_emb1d_2nd(x1d, pos)

        # 第二个通道的注意力机制
        x1d = self.attention_second(x1d)

        if self.layer == 3:
            x1d = self.attention_third(x1d)    # 第三个通道的注意力机制
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)
        else:
            if attention_loss:
                last_attention = x1d
            x1d = self.sigmoid(x1d)

        x1d = F.interpolate(x1d, size=H, mode='linear')    # 反池化
        out = torch.mul(out, x1d.unsqueeze(3))    # 输出=self attention value+input feature

        # 返回注意力值和位置嵌入图
        if return_attention:
            if return_posmap:
                if self.pos_injection == 1:
                    pos_map = (pos_map1)
                elif self.pos_injection == 2:
                    pos_map = (pos_map2)
                return out, x1d, pos_map
            else:
                return out, x1d
        else:
            if attention_loss:
                return out, last_attention
            else:
                return out


# if __name__ == "__main__":