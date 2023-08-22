import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def gamma_correction(image, gamma=1.0):
    inv_gamma = 1 / gamma
    output = np.uint8(((image / 255) ** inv_gamma) * 255)
    return output


def get_image(path, output_path, label, gamma_size):
    imglist = os.listdir(f'{path}/{label}')
    for imgName in imglist:
        output_folder = f'{output_path}/{label}'
        os.makedirs(output_folder, exist_ok=True)

        imgPath = f'{path}/{label}/{imgName}'
        img = cv2.imread(imgPath)

        if label == 'fake':
            img = gamma_correction(img, gamma=gamma_size)

        output = f'{output_folder}/{imgName}'
        cv2.imwrite(output, img)

    return 0


basePath = f'Z:/2nd_paper/dataset/ND/Full/Compare/NestedUVC_DualAttention_Parallel_Fourier_MSE'

# input image dir path

db = ['A', 'B']
gamma_sizes = [0.8, 1.2]

for gamma_size in gamma_sizes:
    for d in db:
        if d == 'A':
            fold = '2-fold'
        else:
            fold = '1-fold'
        iris_B = f'{basePath}/{fold}/{d}'
        iris_B_folders = os.listdir(iris_B)
        print(f'input folder : {iris_B_folders}')

        iris_B_output = f'{basePath}/Attack/Gamma/{d}/gamma_{gamma_size}'
        os.makedirs(iris_B_output, exist_ok=True)

        iris_B_output_folders = os.listdir(iris_B_output)
        print(f'output folder : {iris_B_output_folders}')

        classes = ['live', 'fake']
        for cls in classes:
            get_image(iris_B, iris_B_output, cls, gamma_size)

        #         if input_folder == output_folder:
        #             img_path = f'{iris_B}/{input_folder}'
        #             print(f'now img path : {img_path}')
        #             output_dir = f'{iris_B_output}/{input_folder}'
        #             for i in range(0, len(classes)):
        #                 get_image(img_path, output_dir, classes[i], gamma_size)
        #
        #                 print(f'finish {input_folder} {classes[i]} !!')
        #
        # print(f'------clear {d}------')