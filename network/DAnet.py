from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
import DAHead

class DAnet(nn.Module):
    def __init__(self, num_classes):
        super(DAnet, self).__init__()
        self.ResNet50 = IntermediateLayerGetter(
            resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer4': 'stage4'}
        )
        self.decoder = DAHead.DAHead(in_channels=2048, num_classes=num_classes)

    def forward(self, x):
        feats = self.ResNet50(x)
        # self.ResNet50返回的是一个字典类型的数据.
        x = self.decoder(feats["stage4"])
        return x


if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224).cpu()
    model = DAnet(num_classes=3)
    result = model(x)
    print(result.shape)