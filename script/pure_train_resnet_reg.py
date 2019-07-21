import numpy as np  # linear algebra
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm, trange
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from efficientnet_pytorch import EfficientNet

seed = 42
BATCH_SIZE = 2**6
NUM_WORKERS = 4
LEARNING_RATE = 5e-5
LR_STEP = 3
LR_FACTOR = 0.2
NUM_EPOCHS = 12
LOG_FREQ = 50
TIME_LIMIT = 10 * 60 * 60
RESIZE = 350
WD = 0.003
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"RESIZE: {RESIZE}")
class ImageDataset(Dataset):
    def __init__(self, dataframe, mode):
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        transforms_list = [
            transforms.Resize(RESIZE),
            transforms.CenterCrop(RESIZE)
        ]

        if self.mode == 'train':
            transforms_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(degrees=(0,360), translate=(0.1, 0.1),
                                            scale=(0.8, 1.2),
                                            resample=Image.BILINEAR)
                ])
            ])

        transforms_list.extend([
            transforms.ToTensor(),
        ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        ''' Returns: tuple (sample, target) '''
        filename = self.df['Filename'].values[index]

        directory = '../input/Test' if self.mode == 'test' else '../input/output_combined2'
        sample = Image.open(f'./{directory}/{filename}')

        assert sample.mode == 'RGB'

        image = self.transforms(sample)

        if self.mode == 'test':
            return image
        else:
            return image, self.df['Drscore'].values[index]

    def __len__(self):
        return self.df.shape[0]

def GAP(predicts: torch.Tensor, targets: torch.Tensor):
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    predicts = predicts.cpu().numpy()
    targets = targets.cpu().numpy()

    res = accuracy_score(targets, predicts)
    return res

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n = 1) :
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def train(train_loader, model, criterion, optimizer, epoch, logging = True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()
    model.train()
    num_steps = len(train_loader)

    lr_str = ''

    for i, (input_, target) in enumerate(tqdm(train_loader)):
        if i >= num_steps:
            break

        output = model(input_.float().to(device))
        loss = criterion(output.flatten(), target.float().to(device))

        predicts = torch.round(output.detach())
        predicts[predicts<0] = 0
        predicts[predicts>4] = 4
        avg_score.update(GAP(predicts, target))

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if logging and i % LOG_FREQ == 0:
            print(f'{epoch} [{i}/{num_steps}]\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)
            sys.stdout.flush()
    print(f' * average GAP on train {avg_score.avg:.4f}')
    return avg_score.avg

def train_loop(epochs, train_loader, model, criterion, optimizer,
               validate=True):
    for epoch in trange(1, epochs + 1):
        print(f"learning rate: {lr_scheduler.get_lr()}")
        train(train_loader, model, criterion, optimizer, epoch, logging=True)
        lr_scheduler.step()

    return

labels = pd.read_csv("../input/training-labels.csv")
train_dataset = ImageDataset(labels, mode='train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          drop_last=True,
                          num_workers=NUM_WORKERS)

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
# model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)

model = model.to(device)
model = nn.DataParallel(model)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WD)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP,
                                                   gamma=LR_FACTOR)
global_start_time = time.time()
train_loop(NUM_EPOCHS, train_loader, model, criterion, optimizer)
torch.save(model.state_dict(), sys.argv[1])

os.system("sudo shutdown now")
