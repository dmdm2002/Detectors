import torch
from torchvision import transforms
import numpy as np
from PIL import ImageDraw


# https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/extractor/DeepPixBiS.py
def predict(mask, label, threshold=0.5, score_type='combined'):
    with torch.no_grad():
        if score_type == 'pixel':
            score = torch.mean(mask, axis=(1, 2, 3))
        elif score_type == 'binary':
            score = torch.mean(label, axis=1)
        elif score_type == 'combined':
            score = torch.mean(mask, axis=(1, 2)) + torch.mean(label, axis=1)
        else:
            raise NotImplementedError

        preds = (score > threshold).type(torch.FloatTensor)

        return preds, score


def calc_acc(pred, target):
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()