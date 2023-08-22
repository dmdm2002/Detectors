import pandas as pd
import numpy as np
import os

phases = ['train', 'test']
data = []
for phase in phases:
    if phase == 'train':
        root = 'Z:/2nd_paper/dataset/ND/ROI/Ablation/NestedUVC_DualAttention_Parallel_Fourier_MSE_new/1-fold/A/iris'
    else:
        root = 'Z:/2nd_paper/dataset/ND/ROI/Ablation/NestedUVC_DualAttention_Parallel_Fourier_MSE_new/1-fold/B/iris'

    classes = ['fake', 'live']

    for class_type in classes:
        folder_path = f'{root}/{class_type}'

        images = os.listdir(folder_path)

        for image in images:
            data.append([phase, class_type, image])


data = np.array(data)
data_df = pd.DataFrame(data)
print(data_df)

data_df.to_csv('DNetPAD/Custom/db_csv/NestedUVC_DualAttention_Parallel_Fourier_MSE_new_1fold.csv', index=False)