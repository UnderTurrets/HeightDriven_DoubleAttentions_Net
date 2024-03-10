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
    def __init__(self, num_classes, HAM_num):
        super(HDAnet, self).__init__()
        # 将resnet50分为4个stage，其中stage4是最后一个stage
        self.ResNet50 = IntermediateLayerGetter(
            resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer1': 'stage1', 'layer2': 'stage2', 'layer3': 'stage3', 'layer4': 'stage4'}
        )

        self.HAMlayer_num = HAM_num

        self.DANet_Conv = DAHead(in_channels=2048, num_classes=num_classes)

        if HAM_num >= 5:
            self.HANet_Conv1 = HANet_Conv(in_channel=3, out_channel=256, kernel_size=3,
                                          r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                                          pos_rfactor=8, pooling='mean', dropout_prob=0, pos_noise=0)
        if HAM_num >= 4:
            self.HANet_Conv2 = HANet_Conv(in_channel=256, out_channel=512, kernel_size=3,
                                          r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                                          pos_rfactor=8, pooling='mean', dropout_prob=0, pos_noise=0)
        if HAM_num >= 3:
            self.HANet_Conv3 = HANet_Conv(in_channel=512, out_channel=1024, kernel_size=3,
                                          r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                                          pos_rfactor=8, pooling='mean', dropout_prob=0, pos_noise=0)
        if HAM_num >= 2:
            self.HANet_Conv4 = HANet_Conv(in_channel=1024, out_channel=2048, kernel_size=3,
                                          r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                                          pos_rfactor=8, pooling='mean', dropout_prob=0, pos_noise=0)
        if HAM_num >= 1:
            self.HANet_Conv5 = HANet_Conv(in_channel=2048, out_channel=num_classes, kernel_size=3,
                                          r_factor=64, layer=3, pos_injection=2, is_encoding=1,
                                          pos_rfactor=8, pooling='mean', dropout_prob=0, pos_noise=0)

    def forward(self, x):
        if (self.HAMlayer_num == 1):
            feats = self.ResNet50(x)
            x = feats["stage4"]
            x = self.DANet_Conv(x)
            x = self.HANet_Conv5(feats["stage4"], x)

        elif (self.HAMlayer_num == 2):
            feats = self.ResNet50(x)
            x = self.HANet_Conv4(feats["stage3"], feats["stage4"])
            represent = x
            x = self.DANet_Conv(x)
            x = self.HANet_Conv5(represent, x)

        elif (self.HAMlayer_num == 3):
            feats = self.ResNet50(x)

            x = feats["stage2"]
            x = self.ResNet50['layer3'](x)
            x = self.HANet_Conv3(feats["stage2"], x)

            represent = x
            x = self.ResNet50['layer4'](x)
            x = self.HANet_Conv4(represent, x)

            represent = x
            x = self.DANet_Conv(x)
            x = self.HANet_Conv5(represent, x)

        elif (self.HAMlayer_num == 4):
            feats = self.ResNet50(x)

            x = feats["stage1"]
            x = self.ResNet50['layer2'](x)
            x = self.HANet_Conv2(feats["stage1"], x)

            represent = x
            x = self.ResNet50['layer3'](x)
            x = self.HANet_Conv3(represent, x)

            represent = x
            x = self.ResNet50['layer4'](x)
            x = self.HANet_Conv4(represent, x)

            represent = x
            x = self.DANet_Conv(x)
            x = self.HANet_Conv5(represent, x)

        elif (self.HAMlayer_num == 5):
            feats = self.ResNet50(x)

            represent = x
            x = feats["stage1"]
            x = self.HANet_Conv1(represent, x)

            x = self.ResNet50['layer2'](x)
            x = self.HANet_Conv2(feats["stage1"], x)

            represent = x
            x = self.ResNet50['layer3'](x)
            x = self.HANet_Conv3(represent, x)

            represent = x
            x = self.ResNet50['layer4'](x)
            x = self.HANet_Conv4(represent, x)

            represent = x
            x = self.DANet_Conv(x)
            x = self.HANet_Conv5(represent, x)
        return x



from conf import model_dict
import os
model_list = []
for i in range(1, 6):
    model = HDAnet(num_classes=12, HAM_num=i).cuda()

    # 加载训练好的模型
    model_file = os.path.join(model_dict["HDANet_" + str(i) + "HAM"]["save_path"],
                              model_dict["HDANet_" + str(i) + "HAM"]["model_file"])
    if (os.path.exists(model_file)):
        model.load_state_dict(torch.load(model_file), strict=False)
        print(f"{__name__}:success to load {model_file}")
    else:
        print(f"{__name__}:fail to load {model_file}")

    model_list.append(model)

model_1HAM = model_list[0]
model_2HAM = model_list[1]
model_3HAM = model_list[2]
model_4HAM = model_list[3]
model_5HAM = model_list[4]



if __name__ == "__main__":
    from db.camvid import train_loader,val_loader,test_loader
    from db.camvid import COLORMAP
    from db.camvid import CLASSES
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    # 映射时会产生不知何种原因的毛刺，很难看
    # 使用Cam_COLORMAP创建颜色映射
    # seg_cmap = ListedColormap(Cam_COLORMAP)
    # for index, (img, label) in enumerate(val_loader):
    #
    #     img = img.to(torch.device('cuda:0'))
    #     out_1HAM = model_1HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()
    #     out_5HAM = model_5HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()
    #     img = img.to("cpu")
    #
    #     _, figs = plt.subplots(img.shape[0], 4, figsize=(10, 10))
    #
    #     for i in range(img.shape[0]):
    #         figs[i, 0].imshow(img[i].permute(1, 2, 0))  # 原始图片
    #         figs[i, 0].axes.get_xaxis().set_visible(False)  # 去掉x轴
    #         figs[i, 0].axes.get_yaxis().set_visible(False)  # 去掉y轴
    #
    #         figs[i, 1].imshow(label[i], cmap=ListedColormap(Cam_COLORMAP), vmin=0,
    #                           vmax=len(Cam_CLASSES) - 1)  # Apply colormap to label
    #         figs[i, 1].axes.get_xaxis().set_visible(False)  # 去掉x轴
    #         figs[i, 1].axes.get_yaxis().set_visible(False)  # 去掉y轴
    #
    #         figs[i, 2].imshow(out_1HAM[i], cmap=ListedColormap(Cam_COLORMAP), vmin=0,
    #                           vmax=len(Cam_CLASSES) - 1)  # Apply colormap to label
    #         figs[i, 2].axes.get_xaxis().set_visible(False)  # 去掉x轴
    #         figs[i, 2].axes.get_yaxis().set_visible(False)  # 去掉y轴
    #
    #         figs[i, 3].imshow(out_5HAM[i], cmap=ListedColormap(Cam_COLORMAP), vmin=0,
    #                           vmax=len(Cam_CLASSES) - 1)  # Apply colormap to label
    #         figs[i, 3].axes.get_xaxis().set_visible(False)  # 去掉x轴
    #         figs[i, 3].axes.get_yaxis().set_visible(False)  # 去掉y轴
    #
    #     # 在第一行图片下面添加标题
    #     figs[0, 0].set_title("Image")
    #     figs[0, 1].set_title("Label")
    #     figs[0, 2].set_title("seg1")
    #     figs[0, 3].set_title("seg2")
    #     plt.show()

    for index, (img, label) in enumerate(train_loader):
        img = img.to(torch.device('cuda:0'))
        out_1 = model_1HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()
        out_2 = model_4HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()
        img = img.to("cpu")

        _, figs = plt.subplots(img.shape[0], 4)
        figs[0, 0].set_title("Image")
        figs[0, 1].set_title("Ground-truth")
        figs[0, 2].set_title("seg1")
        figs[0, 3].set_title("seg2")

        for i in range(img.shape[0]):
            # Display original image
            figs[i, 0].imshow(img[i].permute(1, 2, 0))
            figs[i, 0].axis('off')

            # Map mask to colors and display segmented mask with color mapping
            colored_mask = np.zeros((label[i].shape[0], label[i].shape[1], 3), dtype=np.uint8)
            out_1_mask = np.zeros((label[i].shape[0], label[i].shape[1], 3), dtype=np.uint8)
            out_2_mask = np.zeros((label[i].shape[0], label[i].shape[1], 3), dtype=np.uint8)

            for j in range(len(COLORMAP)):
                colored_mask[label[i] == j] = COLORMAP[j]
                out_1_mask[out_1[i] == j] = COLORMAP[j]
                out_2_mask[out_2[i] == j] = COLORMAP[j]

            figs[i, 1].imshow(colored_mask)
            figs[i, 1].axis('off')
            figs[i, 2].imshow(out_1_mask)
            figs[i, 2].axis('off')
            figs[i, 3].imshow(out_2_mask)
            figs[i, 3].axis('off')

        plt.savefig("../res/camvid_segDemo360x480.png",dpi=250)
        plt.show()
