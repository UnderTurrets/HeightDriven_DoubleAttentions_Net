# 导入库
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.data import Dataset, DataLoader
import warnings

import torch
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# 用于做可视化
Cam_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128], [0, 32, 128], [0, 16, 128], [0, 64, 64], [0, 64, 32],
                [0, 64, 16], [64, 64, 128], [0, 32, 16], [32, 32, 32], [16, 16, 16], [32, 16, 128],
                [192, 16, 16], [32, 32, 196], [192, 32, 128], [25, 15, 125], [32, 124, 23], [111, 222, 113],
                ]

# 32类
Cam_CLASSES = ['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child',
               'Column_Pole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter',
               'OtherMoving', 'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol',
               'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel',
               'VegetationMisc', 'Void', 'Wall']

torch.manual_seed(17)

# 自定义数据集CamVidDataset
class CamVidDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(self, images_dir, masks_dir):
        super(CamVidDataset, self).__init__()

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    def __getitem__(self, i):
        # read data

        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('RGB'))

        image = self.transform(image=image, mask=mask)

        return image['image'], image['mask'][:, :, 0]

    def __len__(self):
        return len(self.ids)


# 设置数据集路径
from conf.__init__ import camvid_path
x_train_dir = os.path.join(camvid_path, 'train')
y_train_dir = os.path.join(camvid_path, 'trainannot')
x_valid_dir = os.path.join(camvid_path, 'val')
y_valid_dir = os.path.join(camvid_path, 'valannot')
x_test_dir = os.path.join(camvid_path, 'test')
y_test_dir = os.path.join(camvid_path, 'testannot')

train_dataset = CamVidDataset(
    x_train_dir,
    y_train_dir,
)
val_dataset = CamVidDataset(
    x_valid_dir,
    y_valid_dir,
)
test_dataset = CamVidDataset(
    x_test_dir,
    y_test_dir,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Add this import for color mapping
    from matplotlib.colors import ListedColormap

    for index, (img, label) in enumerate(train_loader):
        # print(label.shape)

        _, figs = plt.subplots(img.shape[0], 2, figsize=(10, 10))

        for i in range(img.shape[0]):
            figs[i, 0].imshow(img[i].permute(1, 2, 0))  # Original image
            figs[i, 0].axes.get_xaxis().set_visible(False)
            figs[i, 0].axes.get_yaxis().set_visible(False)
            figs[i, 1].imshow(label[i], cmap=ListedColormap(Cam_COLORMAP), vmin=0, vmax=len(Cam_CLASSES) - 1)  # Apply colormap to label
            figs[i, 1].axes.get_xaxis().set_visible(False)
            figs[i, 1].axes.get_yaxis().set_visible(False)

        # 在第一行图片下面添加标题
        figs[0, 0].set_title("Image")
        figs[0, 1].set_title("Label")
        plt.show()
        plt.cla()
