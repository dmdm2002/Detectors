import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageFilter

import numpy as np
import glob
import os


class CustomDataset(data.Dataset):
    def __init__(self, dataset_dir, styles, cls, transforms, map_size=14, smoothing=True):
        super(CustomDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.styles = styles
        self.transform = transforms
        self.map_size= map_size
        self.label_weight = 0.99 if smoothing else 1.0

        folder_A = glob.glob(f'{os.path.join(dataset_dir, styles, cls[0])}/*')
        folder_B = glob.glob(f'{os.path.join(dataset_dir, styles, cls[1])}/*')
        self.image_path = []

        for i in range(len(folder_A)):
            self.image_path.append([folder_A[i], 0])

        for i in range(len(folder_B)):
            self.image_path.append([folder_B[i], 1])

    def __getitem__(self, index):
        item = self.transform(Image.open(self.image_path[index][0]).convert('RGB').filter(ImageFilter.EDGE_ENHANCE_MORE))
        label = self.image_path[index][1]

        if label == 0:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)
        else:
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)

        return item, mask, label

    def __len__(self):
        return len(self.image_path)