import os
import argparse
import torchvision.models as models
import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from PBS.eval import calc_acc, predict

from PBS.Loss import PixWiseBCELoss
from PBS.CustomDataset import CustomDataset
from PBS.Model.PBSModel import DeepPixBis

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Description of all argument
device = 'cuda:0'
# Model
model = DeepPixBis()
# model = model.to(device)

outputPath = 'Z:/2nd_paper/backup/Compare/Detectors/Linux/PBSModel/NestedUVC_DualAttention_Parallel_Fourier_MSE/1-fold/CrossFold/try_0/'
datasetPath = 'Z:/2nd_paper/dataset/ND/Full/Compare/NestedUVC_DualAttention_Parallel_Fourier_MSE/Attack/Gamma/'
MaxEpochs = 20

os.makedirs(f'{outputPath}/log', exist_ok=True)
os.makedirs(f'{outputPath}/ckp', exist_ok=True)


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ]
)

labels = ['fake', 'live']
te_dataset = CustomDataset(datasetPath, 'A/gamma_1.2', labels, transform)

lr = 1e-4
loss_fn = PixWiseBCELoss().to(device)
te_loader = data.DataLoader(dataset=te_dataset, batch_size=32, shuffle=False)

checkpoint = torch.load(f'{outputPath}/ckp/4.pth', map_location=device)
model = model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_acc = 0, 0
with torch.no_grad():
    model.eval()
    for _, (item, mask, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Test]')):
        item = item.to(device)
        mask = mask.to(device)
        label = label.to(device)

        net_mask, net_label = model(item)
        loss = loss_fn(net_mask, net_label, mask, label)

        test_loss += loss.item()

        preds, _ = predict(net_mask, net_label, score_type='pixel')
        targets, _ = predict(mask, label, score_type='pixel')
        acc = calc_acc(preds, targets)
        test_acc += acc

test_loss = test_loss / len(te_loader)
test_acc = test_acc / len(te_loader)

print('\n')
print('-------------------------------------------------------------------')
print(f"Test acc: {test_acc} | Test loss: {test_loss}")
print('-------------------------------------------------------------------')