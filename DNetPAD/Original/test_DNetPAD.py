import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from dataset_Loader import datasetLoader
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    device = torch.device('cuda')
    parser.add_argument('-imageFolder', default='Z:/1st/Iris_dataset/nd_labeling_iris_data/Proposed/1-fold/B/iris/',type=str)
    parser.add_argument('-modelPath',  default='Z:/1st/backup/ckp/warsaw/Ablation/DNet/1-fold_5/Logs/DesNet121_22.pth',type=str)
    args = parser.parse_args()
    path = 'Z:/iris_dataset/scores/warsaw/DNetPad/1-fold_6/'
    if not os.path.exists(path):
        os.makedirs(path)

    # dataset = datasetLoader(args.csvPath, args.datasetPath, train_test='test')
    # test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)

    # Load weights of single binary DesNet121 model
    weights = torch.load(args.modelPath)
    DNetPAD = models.densenet121(pretrained=True)
    num_ftrs = DNetPAD.classifier.in_features
    DNetPAD.classifier = nn.Linear(num_ftrs, 2)
    DNetPAD.load_state_dict(weights['state_dict'])
    DNetPAD = DNetPAD.to(device)
    DNetPAD.eval()


    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    imagesScores=[]

    imageFiles_fake = glob.glob(os.path.join(args.imageFolder+'fake','*.png'))
    imageFiles_live = glob.glob(os.path.join(args.imageFolder+'live', '*.png'))
    imageFiles = imageFiles_live + imageFiles_fake
    print(imageFiles)
    # print(imageFiles)
    for imgFile in imageFiles:
            # Read the image
            image = Image.open(imgFile)

            # Image transformation
            tranformImage = transform(image)
            image.close()
            tranformImage = tranformImage.repeat(3, 1, 1) # for NIR images having one channel
            tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
            tranformImage = tranformImage.to(device)

            # Output from single binary CNN model
            start = int(round(time.time() * 1000))
            output = DNetPAD(tranformImage)
            end = int(round(time.time() * 1000))

            print("milli second : ", end - start)
            # PAScore = output.detach().cpu().numpy()[:, 1]
            #
            # # Normalization of output score between [0,1]
            # PAScore = np.minimum(np.maximum((PAScore+15)/35,0),1)
            # imagesScores.append([imgFile, PAScore[0]])
    #
    #
    # # Writing the scores in the csv file
    # with open(os.path.join(path,'Scores_blur_22.csv'),'w',newline='') as fout:
    #     writer = csv.writer(fout)
    #     writer.writerows(imagesScores)