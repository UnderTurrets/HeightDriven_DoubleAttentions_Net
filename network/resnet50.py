from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
import torch

backbone = IntermediateLayerGetter(
    resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True]),
    return_layers={'layer4': 'stage4'}
)
print(backbone)
# test
x = torch.randn(3, 3, 224, 224).cpu()
result = backbone(x)
for k, v in result.items():
    print(k, v.shape)