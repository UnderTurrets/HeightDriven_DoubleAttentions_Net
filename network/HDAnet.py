import random

from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from network.HANet import HANet_Conv


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                    isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


import torch
import torch.nn as nn

gpu = torch.device("cuda")


class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # 创建一个可学习参数a作为权重,并初始化为0.
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        B = self.convB(x)
        C = self.convB(x)
        D = self.convB(x)
        S = self.softmax(torch.matmul(B.view(b, c, h * w).transpose(1, 2), C.view(b, c, h * w)))
        E = torch.matmul(D.view(b, c, h * w), S.transpose(1, 2)).view(b, c, h, w)
        # gamma is a parameter which can be training and iter
        E = self.gamma * E + x

        return E


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        X = self.softmax(torch.matmul(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2)))
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h * w)).view(b, c, h, w)
        X = self.beta * X + x
        return X


class DAHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DAHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, num_classes, kernel_size=3, padding=1, bias=False),
        )

        self.PositionAttention = PositionAttention(in_channels // 4)
        self.ChannelAttention = ChannelAttention()

    def forward(self, x):
        x_PA = self.conv1(x)
        x_CA = self.conv2(x)
        PosionAttentionMap = self.PositionAttention(x_PA)
        ChannelAttentionMap = self.ChannelAttention(x_CA)
        # 这里可以额外分别做PAM和CAM的卷积输出,分别对两个分支做一个上采样和预测,
        # 可以生成一个cam loss和pam loss以及最终融合后的结果的loss.以及做一些可视化工作
        # 这里只输出了最终的融合结果.与原文有一些出入.
        output = self.conv3(PosionAttentionMap + ChannelAttentionMap)
        output = nn.functional.interpolate(output, scale_factor=8, mode="bilinear", align_corners=True)
        output = self.conv4(output)

        return output


class HDAnet(nn.Module):
    def __init__(self, num_classes):
        super(HDAnet, self).__init__()
        self.ResNet50 = IntermediateLayerGetter(
            resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer4': 'stage4'}
        )

        self.DANet_Conv = DAHead(in_channels=2048, num_classes=num_classes)

        self.HANet_Conv = HANet_Conv(in_channel=2048, out_channel=num_classes, kernel_size=3,
                                     r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                                     pos_rfactor=8, pooling='mean', dropout_prob=0, pos_noise=0)

    def forward(self, x):
        feats = self.ResNet50(x)
        # self.ResNet50返回的是一个字典类型的数据.
        x = feats["stage4"]
        represent = x
        x = self.DANet_Conv(x)
        x = self.HANet_Conv(represent, x)

        return x






if __name__ == "__main__":

    import os
    model = HDAnet(num_classes=32)
    if (os.path.exists(r"../checkpoints/HDAnet_50.pth")): model.load_state_dict(
        torch.load(r"../checkpoints/HDAnet_50.pth"), strict=False)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    from db.camvid import train_loader
    import matplotlib.pyplot as plt
    for index,(img,label) in enumerate(train_loader):

        out = model(img).max(1)[1].permute(1, 2, 0).squeeze().cpu().data.numpy()

        _, figs = plt.subplots(img.shape[0], 3, figsize=(10, 10))

        for i in range(img.shape[0]):
            figs[i, 0].imshow(img[i].permute(1, 2, 0))  # 原始图片
            figs[i, 0].axes.get_xaxis().set_visible(False)  # 去掉x轴
            figs[i, 0].axes.get_yaxis().set_visible(False)  # 去掉y轴
            figs[i, 1].imshow(label[i])  # 标签
            figs[i, 1].axes.get_xaxis().set_visible(False)  # 去掉x轴
            figs[i, 1].axes.get_yaxis().set_visible(False)  # 去掉y轴
            figs[i, 2].imshow(out)  # 模型输出结果
            figs[i, 2].axes.get_xaxis().set_visible(False)  # 去掉x轴
            figs[i, 2].axes.get_yaxis().set_visible(False)  # 去掉y轴

        # 在最后一行图片下面添加标题
        figs[img.shape[0] - 1, 0].set_title("Image")
        figs[img.shape[0] - 1, 1].set_title("Label")
        figs[img.shape[0] - 1, 2].set_title("segnet")
        plt.show()
        plt.cla()

