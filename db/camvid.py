# 导入库
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

torch.manual_seed(17)


# 自定义数据集CamVidDataset
class CamVidDataset(torch.utils.data.Dataset):
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
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
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
DATA_DIR = r'dataset\camvid'  # 根据自己的路径来设置
x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')
x_valid_dir = os.path.join(DATA_DIR, 'valid_images')
y_valid_dir = os.path.join(DATA_DIR, 'valid_labels')

train_dataset = CamVidDataset(
    x_train_dir,
    y_train_dir,
)
val_dataset = CamVidDataset(
    x_valid_dir,
    y_valid_dir,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, drop_last=True)