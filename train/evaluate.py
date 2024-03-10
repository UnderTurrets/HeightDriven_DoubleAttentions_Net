import numpy
import numpy as np
import torchvision


class test:    #定义初始函数类
    def __init__(self,a,b,n):        #a为标注图 b为预测图 n为类别数
        k = (a >= 0) & (a < n)       #(a >= 0)防止np.bincount()函数出错   k = (a >= 0) & (a < n) 是一个索引条件
        self.true = a
        self.pred = b
        self.num_class = n
        self.confusion_matrix = np.bincount(self.num_class * self.true[k].astype(int) + self.pred[k], minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)#求混淆矩阵的核心代码


    #PA 像素准确度
    def PA(self):
        pa = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return pa


    #CPA 类别像素准确度
    def CPA(self):
        cpa = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return cpa


    #MPA 平均像素准确度
    def MPA(self):
        mpa = np.nanmean(self.CPA())
        return mpa

    #IOU  交并比
    def IOU(self):
        intersection = np.diag(self.confusion_matrix)#取混淆矩阵对角线元素值
        union = np.sum(self.confusion_matrix,axis=0)+np.sum(self.confusion_matrix,axis=1)-np.diag(self.confusion_matrix)
        iou = intersection/union
        return iou

    #MIOU  平均交并比
    def MIOU(self):
        miou = np.nanmean(self.IOU())
        return miou

    #FWIOU   频率加权交并比
    def FWIOU(self):
        freq = np.sum(self.confusion_matrix,axis=1) / np.sum(self.confusion_matrix)
        iou = self.IOU()
        fwiou = np.sum(freq[freq>0] * iou[freq>0])
        return fwiou


if __name__ =="__main__":
    from db.camvid import test_loader,train_loader,val_loader
    from db.camvid import COLORMAP
    from db.camvid import CLASSES
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    import torch
    from network.HDAnet import model_1HAM,model_2HAM,model_3HAM,model_4HAM,model_5HAM





    total_PA = 0
    average_PA = 0
    total_FWIOU = 0
    average_FWIOU = 0
    num = 0
    for index, (img, label) in enumerate(test_loader):
        img = img.to(torch.device('cuda:0'))
        out_1 = model_1HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()
        # out_2 = model_5HAM(img).max(dim=1)[1].squeeze(dim=1).cpu().data.numpy()
        img = img.to("cpu")

        _, figs = plt.subplots(img.shape[0], 3)
        figs[0, 0].set_title("Image")
        figs[0, 1].set_title("Ground-truth")
        figs[0, 2].set_title("ours")
        # figs[0, 3].set_title("seg2")

        unique_values = np.unique(label)
        num_unique_values = len(unique_values)
        print("Number of unique values in labels:", num_unique_values)

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
                # out_2_mask[out_2[i] == j] = COLORMAP[j]

            # figs[i, 1].imshow(colored_mask)
            # figs[i, 1].axis('off')
            # figs[i, 2].imshow(out_1_mask)
            # figs[i, 2].axis('off')
            # figs[i, 3].imshow(out_2_mask)
            # figs[i, 3].axis('off')

            TEST = test(label[i].numpy(), out_1[i], num_unique_values)

            total_PA +=TEST.PA()
            total_FWIOU += TEST.FWIOU()
            num += 1
            average_PA = total_PA/num
            average_FWIOU = total_FWIOU/num




            # print(f'混淆矩阵:{TEST.confusion_matrix}')
            print(f'此次像素准确度:{TEST.PA()}')
            print(f'平均像素准确度:{average_PA}')

            # print(f'类别像素准确度{TEST.CPA()}')
            # print(f'平均像素准确度:{TEST.MPA()}')
            # print(f'交并比:{TEST.IOU()}')
            # print(f'平均交并比:{TEST.MIOU()}')

            print(f'此次频率加权交并比:{TEST.FWIOU()}')
            print(f'平均频率加权交并比:{average_FWIOU}')

        # plt.show()



