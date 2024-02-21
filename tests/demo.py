if __name__ == "__main__":

    from network.HDAnet import model_5HAM
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    from PIL import Image

    # 测试图片路径
    img_path = r"D:\Datasets\cityscapes\leftImg8bit_trainvaltest\val\lindau\lindau_000000_000019_leftImg8bit.png"
    img = np.array(Image.open(img_path).convert('RGB'))

    # 颜色映射
    from db.camvid import COLORMAP as Camvid_COLORMAP


    # 使用transforms将图像转换成合适的形状和通道顺序
    import albumentations
    from albumentations.pytorch.transforms import ToTensorV2
    transform = albumentations.Compose([
        albumentations.Resize(360, 480),  # 调整大小

        albumentations.Normalize(),
        ToTensorV2(),  # 转换成张量
    ])
    img = transform(image=img)["image"].unsqueeze(0)  # 增加维度 batch_size

    # import torchvision.transforms as transforms
    # transform = transforms.Compose([
    #
    #
    #     # A.HorizontalFlip(),
    #     # A.RandomBrightnessContrast(),
    #     # A.RandomSnow(),
    #
    #     transforms.ToTensor(),
    #     transforms.Resize(size=(500, 1000)),
    #     transforms.Normalize(
    #         mean=(0.485, 0.456, 0.406),
    #         std=(0.229, 0.224, 0.225),
    #     ),
    # ])
    # img = transform(img).unsqueeze(0)  # 增加维度 batch_size

    img = img.to(torch.device('cuda:0'))
    out = model_5HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()

    img = img.to('cpu')

    _, figs = plt.subplots(1, 2,figsize=(10,10))
    # 添加标题
    figs[0].set_title("Image")
    figs[1].set_title("segnet")

    figs[0].imshow(img[0].permute(1, 2, 0))  # 原始图片
    figs[0].axis('off')

    colored_mask = np.zeros((out[0].shape[0], out[0].shape[1], 3), dtype=np.uint8)
    for j in range(len(Camvid_COLORMAP)):
        colored_mask[out[0] == j] = Camvid_COLORMAP[j]
    figs[1].imshow(colored_mask)  # Apply colormap to label
    figs[1].axis('off')

    plt.show()
