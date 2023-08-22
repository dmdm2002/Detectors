import argparse
import torchvision.models as models
import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
from DNetPAD.Custom.CustomDataset import CustomDataset
from torchmetrics.classification import ConfusionMatrix

# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--MaxEpochs', type=int, default=30)
# 1-fold: Gan(1-fold)B -> Gan(2-fold)A, 2-fold: Gan(2-fold)A->Gan(1-fold)B
parser.add_argument('--datasetPath', required=False, default='Z:/2nd_paper/dataset/ND/ROI/Compare/PGGAN/Attack/Gamma/A', type=str)
parser.add_argument('--outputPath', required=False, default='Z:/2nd_paper/backup/Compare/Detectors/Linux/DNet-PAD/PGGAN/CrossFold/1-fold/try_5', type=str)
parser.add_argument('--method', default='DesNet121', type=str)
parser.add_argument('--nClasses', default=2, type=int)

device = 'cuda'
args = parser.parse_args()

# Model
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, args.nClasses)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ]
)
labels = ['fake', 'live']
te_dataset = CustomDataset(args.datasetPath, 'gamma_0.8', labels, transform)
# te_dataset = CustomDataset(args.datasetPath, 'gamma_1.2', labels, transform)
criterion = nn.CrossEntropyLoss().to(device)

te_loader = data.DataLoader(dataset=te_dataset, batch_size=args.batchSize, shuffle=False)

test_loss, test_acc = 0, 0
tp, tn, fp, fn = 0, 0, 0, 0

confmat = ConfusionMatrix(task="binary", num_classes=2)

checkpoint = torch.load(f'{args.outputPath}/ckp/23.pth', map_location=device)
model = model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])

with torch.no_grad():
    model.eval()
    for _, (item, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Attack Image Test]')):
        item = item.to(device)
        label = label.to(device)

        logits = model(item)
        loss = criterion(logits, label)

        test_loss += loss.item()
        test_acc += (logits.argmax(1) == label).type(torch.float).sum().item()

        confmat = ConfusionMatrix(task="binary", num_classes=2)
        [tp_batch, fn_batch], [fp_batch, tn_batch] = confmat(logits.argmax(1).cpu(), label.cpu())
        tp += tp_batch
        tn += tn_batch
        fp += fp_batch
        fn += fn_batch

test_loss = test_loss / len(te_loader)
test_acc = test_acc / len(te_loader.dataset)
apcer = fp / (tn + fp) if (tn + fp) != 0 else 0
bpcer = fn / (fn + tp) if (fn + tp) != 0 else 0
acer = (apcer + bpcer) / 2

print('\n')
print('-------------------------------------------------------------------')
print(f"||Test acc: {test_acc} | Test loss: {test_loss} ||")
print(f'||APCER: {apcer * 100}  |  BPCER: {bpcer * 100}  |  ACER: {acer * 100}||')
print('-------------------------------------------------------------------')