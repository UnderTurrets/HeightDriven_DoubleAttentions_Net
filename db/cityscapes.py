import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import glob


COLORMAP = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [0, 0, 0], [111, 74,  0], [81,  0, 81], [128, 64,128],
            [244, 35,232], [250,170,160], [230,150,140], [70, 70, 70],
            [102,102,156], [190,153,153], [180,165,180], [150,100,100],
            [150,120, 90], [153,153,153], [153,153,153], [250,170, 30],
            [220,220,  0], [107,142, 35], [152,251,152], [70,130,180], [220, 20, 60],
            [255,  0,  0], [0,  0,142], [0,  0, 70], [0, 60,100], [0,  0, 90], [0,  0,110],
            [0, 80,100], [0,  0,230], [119, 11, 32]
            ]


CLASSES = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground',
               'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail',
               'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
               'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
               'train', 'motorcycle', 'bicycle']


torch.manual_seed(17)


# 自定义数据集
class CityScapesDataset(torch.utils.data.Dataset):
    """CityScapes Dataset. Read images, apply augmentation and preprocessing transformations.

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
            A.Resize(360, 480),

            # A.HorizontalFlip(),
            # A.RandomBrightnessContrast(),
            # A.RandomSnow(),

            A.Normalize(),
            ToTensorV2(),
        ])

        self.images_fps = sorted(glob.glob(os.path.join(images_dir, '*', '*.png')))
        self.masks_fps = sorted(glob.glob(os.path.join(masks_dir, '*', '*_labelIds.png')))

        assert len(self.images_fps) == len(self.masks_fps)

    def __getitem__(self, i):
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('RGB'))
        image = self.transform(image=image, mask=mask)

        return image['image'], image['mask'][:, :, 0]

    def __len__(self):
        return len(self.images_fps)


# 设置数据集路径
from conf.__init__ import cityscapes_path

train_dataset = CityScapesDataset(
    os.path.join(cityscapes_path, 'leftImg8bit_trainvaltest', 'train'),
    os.path.join(cityscapes_path, 'gtFine_trainvaltest', 'train'),
)
val_dataset = CityScapesDataset(
    os.path.join(cityscapes_path, 'leftImg8bit_trainvaltest', 'val'),
    os.path.join(cityscapes_path, 'gtFine_trainvaltest', 'val'),
)

test_dataset = CityScapesDataset(
    os.path.join(cityscapes_path, 'leftImg8bit_trainvaltest', 'test'),
    os.path.join(cityscapes_path, 'gtFine_trainvaltest', 'test'),
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    for index, (img, label) in enumerate(train_loader):
        _, figs = plt.subplots(img.shape[0], 2, figsize=(10, 10))
        figs[0, 0].set_title("Image")
        figs[0, 1].set_title("Label")

        for i in range(img.shape[0]):
            # Display original image
            figs[i, 0].imshow(img[i].permute(1, 2, 0))
            figs[i, 0].axis('off')

            # Map mask to colors and display segmented mask with color mapping
            colored_mask = np.zeros((label[i].shape[0], label[i].shape[1], 3), dtype=np.uint8)
            for j in range(len(COLORMAP)):
                colored_mask[label[i] == j] = COLORMAP[j]
                # plt.imshow(colored_mask)
                # plt.show()

            figs[i, 1].imshow(colored_mask)
            figs[i, 1].axis('off')

        plt.show()
