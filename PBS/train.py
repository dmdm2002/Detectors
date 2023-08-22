import os
import argparse
import torchvision.models as models
import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from PBS.eval import calc_acc, predict
from PBS.Loss import PixWiseBCELoss
from PBS.CustomDataset import CustomDataset
from PBS.Model.PBSModel import DeepPixBis
from PBS.Model.APBSModel import APBSModel

# Description of all argument
device = 'cuda:0'
# Model
model = DeepPixBis()
model = model.to(device)

outputPath = 'Z:/2nd_paper/backup/Compare/Detectors/Linux/PBSModel/NestedUVC_DualAttention_Parallel_Fourier_MSE/1-fold/CrossFold/try_0/'
tr_datasetPath = 'Z:/2nd_paper/dataset/ND/Full/Compare/NestedUVC_DualAttention_Parallel_Fourier_MSE/1-fold'
te_datasetPath = 'Z:/2nd_paper/dataset/ND/Full/Compare/NestedUVC_DualAttention_Parallel_Fourier_MSE/2-fold'
MaxEpochs = 10

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
tr_dataset = CustomDataset(tr_datasetPath, 'B', labels, transform)
te_dataset = CustomDataset(te_datasetPath, 'A', labels, transform)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = PixWiseBCELoss().to(device)

tr_loader = data.DataLoader(dataset=tr_dataset, batch_size=24, shuffle=True)
te_loader = data.DataLoader(dataset=te_dataset, batch_size=24, shuffle=False)

summary = SummaryWriter(f'{outputPath}/log')

best_score = {'epoch': 0, 'acc': 0, 'loss': 0}
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
for ep in range(MaxEpochs):
    train_loss, train_acc = 0, 0
    test_loss, test_acc = 0, 0

    model.train()
    for _, (item, mask, label) in enumerate(tqdm.tqdm(tr_loader, desc=f'[Train {ep}/{MaxEpochs}]')):
        item = item.to(device)
        mask = mask.to(device)
        label = label.to(device)

        net_mask, net_label = model(item)
        loss = loss_fn(net_mask, net_label, mask, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate predictions
        preds, _ = predict(net_mask, net_label, score_type='pixel')
        targets, _ = predict(mask, label, score_type='pixel')
        acc = calc_acc(preds, targets)
        train_acc += acc

        # probs = net_label >= torch.FloatTensor([0.5]).to(device)
        # train_acc += (probs == label).sum().item()

    with torch.no_grad():
        model.eval()
        for _, (item, mask, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Test {ep}/{MaxEpochs}]')):
            item = item.to(device)
            mask = mask.to(device)
            label = label.to(device)

            net_mask, net_label = model(item)
            loss = loss_fn(net_mask, net_label, mask, label)

            test_loss += loss.item()

            # Calculate predictions
            preds, _ = predict(net_mask, net_label, score_type='pixel')
            targets, _ = predict(mask, label, score_type='pixel')
            acc = calc_acc(preds, targets)
            test_acc += acc

    train_loss = train_loss / len(tr_loader)
    train_acc = train_acc / len(tr_loader)

    test_loss = test_loss / len(te_loader)
    test_acc = test_acc / len(te_loader)

    if best_score['acc'] <= test_acc:
        best_score['acc'] = test_acc
        best_score['epoch'] = ep
        best_score['loss'] = test_loss

    print('\n')
    print('-------------------------------------------------------------------')
    print(f"Epoch: {ep}/{MaxEpochs}")
    print(f"Train acc: {train_acc} | Train loss: {train_loss}")
    print(f"Test acc: {test_acc} | Test loss: {test_loss}")
    print('-------------------------------------------------------------------')
    print(f"Best acc epoch: {best_score['epoch']}")
    print(f"Best acc: {best_score['acc']} | Best loss: {best_score['loss']}")
    print('-------------------------------------------------------------------')

    summary.add_scalar('Train/acc', train_loss, ep)
    summary.add_scalar('Train/loss', train_acc, ep)
    summary.add_scalar('Test/acc', test_acc, ep)
    summary.add_scalar('Test/loss', test_loss, ep)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_adamw_state_dict": optimizer.state_dict(),
            "epoch": ep,
        },
        os.path.join(f"{outputPath}/ckp", f"{ep}.pth"),
    )

    if best_score['acc'] == 1.0:
        break
