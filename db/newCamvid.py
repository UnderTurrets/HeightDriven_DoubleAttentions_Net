from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tfs
im = Image.open(r"D:\Desktop\Datasets\camvid\test\0001TP_008550.png")
lab = Image.open(r"D:\Desktop\Datasets\camvid\testannot\0001TP_008550.png")
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.imshow(lab)
lab = np.array(lab)
print(lab.shape)
lab = torch.from_numpy(lab)
print(lab.shape)
im = tfs.ToTensor()(im)
print(im.shape)
plt.show()
