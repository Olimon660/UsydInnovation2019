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
BATCH_SIZE = 2**4
NUM_WORKERS = 6
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
LOG_FREQ = 50
TIME_LIMIT = 10 * 60 * 60
RESIZE = 512
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                            scale=(0.8, 1.2),
                                            resample=Image.BILINEAR)
                ])
            ])

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
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

def train(train_loader, model, criterion, optimizer, epoch, logging = True):
    model.train()
    num_steps = len(train_loader)

    lr_str = ''

    for i, (input_, target) in tqdm(enumerate(train_loader)):
        if i >= num_steps:
            break

        output = model(input_.to(device))
        loss = criterion(output, target.to(device))

        confs, predicts = torch.max(output.detach(), dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return

def train_loop(epochs, train_loader, model, criterion, optimizer,
               validate=True):
    for epoch in trange(1, epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch, logging=True)

    return

labels = pd.read_csv("../input/training-labels.csv")
train_dataset = ImageDataset(labels, mode='train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          drop_last=True,
                          num_workers=NUM_WORKERS)

model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)

model = model.to(device)
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
global_start_time = time.time()
train_loop(NUM_EPOCHS, train_loader, model, criterion, optimizer)
torch.save(model.state_dict(), sys.argv[1])
