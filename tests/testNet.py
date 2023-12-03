if __name__ == "__main__":

    from network.HDAnet import HDAnet
    import torch
    import os
    model = HDAnet(num_classes=32)
    if (os.path.exists(r"D:\Desktop\scholarly achievements\2023大创\checkpoints\model_oneHANet\HDAnet_50.pth")):
        model.load_state_dict(
            torch.load(r"D:\Desktop\scholarly achievements\2023大创\checkpoints\model_oneHANet\HDAnet_50.pth"),
            strict=False)


    from db.camvid import Cam_COLORMAP, Cam_CLASSES
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap


    # 使用Cam_COLORMAP创建颜色映射
    seg_cmap = ListedColormap(Cam_COLORMAP)
    from PIL import Image

    # 下载的测试图片路径
    img_path = r"D:\Desktop\th.jpg"
    img = Image.open(img_path)

    from torchvision import transforms

    # 使用transforms将图像转换成合适的形状和通道顺序
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小，根据你的网络输入大小调整
        transforms.ToTensor(),  # 转换成张量
    ])
    img = transform(img).unsqueeze(0)  # 增加一个批次维度
    out = model(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()

    _, figs = plt.subplots(1, 2, figsize=(10, 10))

    figs[0].imshow(img[0].permute(1, 2, 0))  # 原始图片
    figs[0].axes.get_xaxis().set_visible(False)  # 去掉x轴
    figs[0].axes.get_yaxis().set_visible(False)  # 去掉y轴

    figs[1].imshow(out[0], cmap=ListedColormap(Cam_COLORMAP), vmin=0,
                   vmax=len(Cam_CLASSES) - 1)  # Apply colormap to label
    figs[1].axes.get_xaxis().set_visible(False)  # 去掉x轴
    figs[1].axes.get_yaxis().set_visible(False)  # 去掉y轴

    # 在最后一行图片下面添加标题
    figs[0].set_title("Image")
    figs[1].set_title("segnet")
    plt.show()
    plt.cla()
