import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageFilter

import glob
import os


class CustomDataset(data.Dataset):
    def __init__(self, dataset_dir, styles, cls, transforms):
        super(CustomDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.styles = styles
        self.transform = transforms

        folder_A = glob.glob(f'{os.path.join(dataset_dir, styles, "iris", cls[0])}/*')
        folder_B = glob.glob(f'{os.path.join(dataset_dir, styles, "iris", cls[1])}/*')
        self.image_path = []

        for i in range(len(folder_A)):
            self.image_path.append([folder_A[i], 0])

        for i in range(len(folder_B)):
            self.image_path.append([folder_B[i], 1])

    def __getitem__(self, index):
        item = self.transform(Image.open(self.image_path[index][0]).convert('RGB').filter(ImageFilter.EDGE_ENHANCE_MORE))
        label = self.image_path[index][1]

        return [item, label]

    def __len__(self):
        return len(self.image_path)