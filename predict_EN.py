import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from PIL import ImageFile
import pandas as pd
import torchvision.transforms.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
from efficientnet_pytorch import EfficientNet

BATCH_SIZE = 2**6
NUM_WORKERS = 1
RESIZE = 350

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("test_file", help='path to the test.csv. e.g. ./input/SampleSubmission.csv')
parser.add_argument("model", help='path to the model to be used. e.g. ./model/final.ptm')
parser.add_argument("out", help='path to the output file. e.g. ./submission.csv')
args = parser.parse_args()

class ImageDataset(Dataset):
    def __init__(self, dataframe, mode, hflip = False, vflip = False):
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode
        self.hflip = hflip
        self.vflip = vflip

        transforms_list = [
            transforms.CenterCrop(RESIZE),
            transforms.ToTensor(),
        ]

        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        ''' Returns: tuple (sample, target) '''
        filename = self.df['Id'].values[index]

        directory = './input/Test'
        sample = Image.open(f'./{directory}/gb_12_{filename}')
        if self.hflip:
            sample = F.hflip(sample)
        if self.vflip:
            sample = F.vflip(sample)
        image = self.transforms(sample)

        if self.mode == 'test':
            return image
        else:
            return image, self.df['Drscore'].values[index]

    def __len__(self):
        return self.df.shape[0]

def inference(data_loader, model):
    ''' Returns predictions and targets, if any. '''
    model.eval()

    all_predicts, all_targets = [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            if data_loader.dataset.mode != 'test':
                input_, target = data
            else:
                input_, target = data, None

            output = model(input_.float().to(device))
            all_predicts.append(output)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, targets

labels = pd.read_csv(args.test_file)
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)

model = model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.model))

test_dataset = ImageDataset(labels, mode='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

predicts, targets = inference(test_loader, model)

test_dataset = ImageDataset(labels, mode='test', hflip=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

predicts_hf, targets = inference(test_loader, model)

test_dataset = ImageDataset(labels, mode='test', vflip=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

predicts_vf, targets = inference(test_loader, model)

test_dataset = ImageDataset(labels, mode='test', hflip=True, vflip=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

predicts_hf_vf, targets = inference(test_loader, model)

predicts_all = np.asarray(
    [predicts.cpu().numpy().flatten(),
     predicts_hf.cpu().numpy().flatten(),
     predicts_vf.cpu().numpy().flatten(),
    predicts_hf_vf.cpu().numpy().flatten()])

predicts_all = np.mean(predicts_all, axis=0)

predicts_all = np.round(predicts_all, 0).astype(int)

predicts_all[predicts_all > 4] = 4
predicts_all[predicts_all < 0] = 0

labels['Expected'] = np.abs(predicts_all)
labels.to_csv(args.out, index=False)
