import torch
from torch import nn
import torch.nn.functional as F


class PixWiseBCELoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.criterion = nn.BCELoss()
        # self.map_criterion = nn.SmoothL1Loss()
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        loss_pixel_map = self.criterion(net_mask.squeeze(-1), target_mask.squeeze(-1))
        loss_bce  = self.criterion(net_label.type(torch.FloatTensor).squeeze(-1), target_label.type(torch.FloatTensor).squeeze(-1))
        loss = self.beta * loss_bce + (1 - self.beta) * loss_pixel_map
        return loss

