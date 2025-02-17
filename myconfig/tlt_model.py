
from mmpretrain import get_model
from mmpretrain.registry import MODELS
import torch
import torch.nn as nn

from timm.models.registry import register_model


@register_model
def ci2pvit_imagenet(pretrained=False, **kwargs):
    model = get_model('./config_ci2pvit_imagenet.py', pretrained='../../checkpoint/ci2pvit_imagenet/imagenet_epoch_385.pth')
    model = model.backbone
    return model

def test():
    m = ci2pvit_imagenet()
    print(m)
    m.train()
    img = torch.randn(1, 3, 256, 256)
    out = m(img)
    if isinstance(out, tuple):
        print(len(out), out[0].shape, out[1].shape, out[2])
    else:
        print(out.shape)

if __name__=='__main__':
    test()