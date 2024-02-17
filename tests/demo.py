if __name__ == "__main__":

    from network.HDAnet import model_1HAM, model_2HAM, model_3HAM, model_4HAM, model_5HAM
    import matplotlib.pyplot as plt
    import torch

    from PIL import Image

    # 下载的测试图片路径
    img_path = r"C:\Users\Xu Han\Desktop\R-C.jpg"
    img = Image.open(img_path)

    from torchvision import transforms

    # 使用transforms将图像转换成合适的形状和通道顺序
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小
        transforms.ToTensor(),  # 转换成张量
    ])

    img = transform(img).unsqueeze(0)  # 增加维度 batch_size
    img = img.to(torch.device('cuda:0'))
    out = model_1HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()
    img = img.to('cpu')

    _, figs = plt.subplots(1, 2,)

    figs[0].imshow(img[0,:,:,:].moveaxis(0, 2))  # 原始图片
    figs[0].axes.get_xaxis().set_visible(False)  # 去掉x轴
    figs[0].axes.get_yaxis().set_visible(False)  # 去掉y轴

    figs[1].imshow(out[0,:,:])  # Apply colormap to label
    figs[1].axes.get_xaxis().set_visible(False)  # 去掉x轴
    figs[1].axes.get_yaxis().set_visible(False)  # 去掉y轴

    # 在第一行图片下面添加标题
    figs[0].set_title("Image")
    figs[1].set_title("segnet")
    plt.show()
    plt.cla()
