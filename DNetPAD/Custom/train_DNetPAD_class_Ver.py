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
from DNetPAD.Custom.CustomDataset import CustomDataset
from torchmetrics.classification import ConfusionMatrix


class TrDNet(object):
    def __init__(self, tr_path, te_path, Gmodel, fold):
        super(TrDNet, self).__init__()
        self.fold = fold
        self.batchSize = 2
        self.MaxEpochs = 50
        self.Tr_datasetPath = tr_path
        self.Te_datasetPath = te_path
        self.outputPath = f'Z:/2nd_paper/backup/Compare/Detectors/Linux/DNet-PAD/{Gmodel}/CrossFold/{self.fold}/try_6/'
        self.method = 'DesNet121'
        self.nClasses = 2

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, self.nClasses)
        model = model.to(self.device)

        os.makedirs(f'{self.outputPath}/log', exist_ok=True)
        os.makedirs(f'{self.outputPath}/ckp', exist_ok=True)

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ]
        )
        labels = ['fake', 'live']
        if self.fold == '1-fold':
            tr_dataset = CustomDataset(self.Tr_datasetPath, 'B', labels, transform)
            te_dataset = CustomDataset(self.Te_datasetPath, 'A', labels, transform)
        else:
            tr_dataset = CustomDataset(self.Tr_datasetPath, 'A', labels, transform)
            te_dataset = CustomDataset(self.Te_datasetPath, 'B', labels, transform)

        lr = 0.005
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
        lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        tr_loader = data.DataLoader(dataset=tr_dataset, batch_size=self.batchSize, shuffle=True)
        te_loader = data.DataLoader(dataset=te_dataset, batch_size=self.batchSize, shuffle=False)

        summary = SummaryWriter(f'{self.outputPath}/log')

        best_score = {'epoch': 0, 'acc': 0, 'apcer': 0, 'bpcer': 0, 'acer': 0}
        confmat = ConfusionMatrix(task="binary", num_classes=2)

        for ep in range(self.MaxEpochs):
            train_loss, train_acc = 0, 0
            test_loss, test_acc = 0, 0
            tp, tn, fp, fn = 0, 0, 0, 0

            model.train()
            for _, (item, label) in enumerate(tqdm.tqdm(tr_loader, desc=f'[Train {ep}/{self.MaxEpochs}]')):
                item = item.to(self.device)
                label = label.to(self.device)

                logits = model(item)
                loss = criterion(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += (logits.argmax(1) == label).type(torch.float).sum().item()

            with torch.no_grad():
                model.eval()
                for _, (item, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Test {ep}/{self.MaxEpochs}]')):
                    item = item.to(self.device)
                    label = label.to(self.device)

                    logits = model(item)
                    loss = criterion(logits, label)

                    [tp_batch, fn_batch], [fp_batch, tn_batch] = confmat(logits.argmax(1).cpu(), label.cpu())
                    tp += tp_batch
                    tn += tn_batch
                    fp += fp_batch
                    fn += fn_batch

                    test_loss += loss.item()
                    test_acc += (logits.argmax(1) == label).type(torch.float).sum().item()

            lr_sched.step(ep)

            train_loss = train_loss / len(tr_loader)
            train_acc = train_acc / len(tr_loader.dataset)

            test_loss = test_loss / len(te_loader)
            test_acc = test_acc / len(te_loader.dataset)

            apcer = fp / (tn + fp) if (tn + fp) != 0 else 0
            bpcer = fn / (fn + tp) if (fn + tp) != 0 else 0
            acer = (apcer + bpcer) / 2

            if best_score['acc'] <= test_acc:
                best_score['acc'] = test_acc
                best_score['epoch'] = ep
                best_score['apcer'] = apcer.item() * 100
                best_score['bpcer'] = bpcer.item() * 100
                best_score['acer'] = acer.item() * 100

            print('\n')
            print('-------------------------------------------------------------------')
            print(f"Epoch: {ep}/{self.MaxEpochs}")
            print(f"Train acc: {train_acc} | Train loss: {train_loss}")
            print(f"Test acc: {test_acc} | Test loss: {test_loss}")
            print(f'APCER: {apcer * 100}  |  BPCER: {bpcer * 100}  |  ACER: {acer * 100}')
            print('-------------------------------------------------------------------')
            print(f"Best acc epoch: {best_score['epoch']}")
            print(f"Best acc: {best_score['acc']}")
            print(
                f"APCER: {best_score['apcer']}  |  BPCER: {best_score['bpcer']}  |  ACER: {best_score['acer']}")
            print('-------------------------------------------------------------------')

            summary.add_scalar('Train/acc', train_loss, ep)
            summary.add_scalar('Train/loss', train_acc, ep)

            summary.add_scalar('Test/acc', test_acc, ep)
            summary.add_scalar('Test/loss', test_loss, ep)

            summary.add_scalar('Test/apcer', apcer, ep)
            summary.add_scalar('Test/bpcer', bpcer, ep)
            summary.add_scalar('Test/acer', acer, ep)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_adamw_state_dict": optimizer.state_dict(),
                    "epoch": ep,
                },
                os.path.join(f"{self.outputPath}/ckp", f"{ep}.pth"),
            )

        return best_score
