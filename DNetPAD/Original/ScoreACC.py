import pandas as pd
import numpy as np
import os
import re

predict_score_csv = pd.read_csv('E:/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/2-fold/A_blur/iris/Scores.csv')

phases = ['test']
data = []
for phase in phases:
    if phase == 'train':
        root = 'E:/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/2-fold/A_blur/iris'
    else:
        root = 'E:/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/2-fold/A_blur/iris'

    classes = ['fake', 'live']

    for class_type in classes:
        folder_path = f'{root}/{class_type}'

        images = os.listdir(folder_path)

        class_num = 1
        if class_type == 'fake':
            class_num = 0

        for image in images:
            data.append([class_num, image])


data = np.array(data)
print(data)
# data = pd.DataFrame(data)
#
# print(predict_score_csv)
predict_list = np.array(predict_score_csv)
print(predict_list)
TrueCnt = 0
FalseCnt = 0

for predict in predict_list:
    path = predict[0]
    prob = predict[1]

    image_name_split = re.split("\\\\", path)
    image_name = image_name_split[1]

    if prob > 0.5:
        predict_label = 1
    else:
        predict_label = 0

    for value in data:
        name = value[1]
        label = value[0]

        if name == image_name:
            if int(label) == int(predict_label):
                TrueCnt = TrueCnt + 1
            else:
                FalseCnt = FalseCnt + 1

acc = TrueCnt/len(data)

print(acc)