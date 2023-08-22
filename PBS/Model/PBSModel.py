import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


"""https://github.com/voqtuyen/deep-pix-bis-pad.pytorch/blob/master/trainer/base.py"""


class DeepPixBis(nn.Module):
    """
    The class defining Deep Pixel-wise Binary Supervision for Face Presentation Attack
    """
    def __init__(self, pretrained=False):
        super(DeepPixBis, self).__init__()
        dense = models.densenet161(pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[0:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14*14, 1)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        dec = self.linear(out_map.view(-1, 14*14))
        dec = F.sigmoid(dec)
        return out_map, dec
