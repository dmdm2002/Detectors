import torch
import torch.nn as nn
from torchvision import models

from PBS.Model.Attention import SpatialAttention


class APBSModel(nn.Module):
    def __init__(self):
        super(APBSModel, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', '161', pretrained=True)
        self.inc = self.input_conv()
        self.inc_attn = SpatialAttention()
        self.inc_comp_attn = nn.MaxPool2d(kernel_size=(5, 5), stride=5)

        self.dense_block_1 = self.backbone.features.denseblock1
        self.transition1 = self.backbone.features.transition1
        self.dense_attn_1 = SpatialAttention()
        self.comp_attn_1 = nn.MaxPool2d(kernel_size=(5, 5), stride=1)

        self.dense_block_2 = self.backbone.features.denseblock2
        self.transition2 = self.backbone.features.transition2
        self.dense_attn_2 = SpatialAttention()

        self.conv_3x3 = nn.Conv2d(64+128+256, 384, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_1x1 = nn.Conv2d(384, 1, kernel_size=(1, 1), stride=1, padding=0)
        self.linear = nn.Linear(14*14, 1)

    def input_conv(self):
        return nn.Sequential(
            self.backbone.features.conv0,
            self.backbone.features.norm0,
            self.backbone.features.relu0,
            self.backbone.features.pool0,
        )

    def forward(self, x):
        x_inc = self.inc(x)
        x_inc_attn = self.inc_attn(x_inc)
        x_inc_attn = torch.mul(x_inc, x_inc_attn)

        x_dense_1 = self.dense_block_1(x_inc_attn)
        x_transition_1 = self.transition1(x_dense_1)
        x_dense_attn_1 = self.dense_attn_1(x_transition_1)
        x_dense_attn_1 = torch.mul(x_transition_1, x_dense_attn_1)

        x_dense_2 = self.dense_block_2(x_dense_attn_1)
        x_transition_2 = self.transition2(x_dense_2)
        x_dense_attn_2 = self.dense_attn_2(x_transition_2)
        x_dense_attn_2 = torch.mul(x_transition_2, x_dense_attn_2)

        x_comp_attn_inc = self.inc_comp_attn(x_inc_attn)
        x_comp_attn_1 = self.comp_attn_1(x_dense_attn_1)

        print(x_inc_attn.shape)
        print(x_dense_attn_1.shape)
        print(x_dense_attn_2.shape)
        print('--------------------------------------------------')
        print(x_comp_attn_inc.shape)
        print(x_comp_attn_1.shape)
        print(x_dense_attn_2.shape)
        x_cat = torch.cat([x_comp_attn_inc, x_comp_attn_1, x_dense_attn_2], dim=1)

        x_out = self.conv_1x1(x_cat)
        out_map = nn.functional.sigmoid(x_out)
        out = self.linear(out_map.view(-1, 14*14))
        out = nn.functional.sigmoid(out)
        out = torch.flatten(out)

        return out_map, out

