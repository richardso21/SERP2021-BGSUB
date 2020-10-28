import h5py, os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# make results reproducable
torch.manual_seed(42)

# MODEL & HELPERS >>>

def accuracy(outs, labels):
    res = {}
    # check accuracy for 3 different thresholds
    for th in [.50, .75, .80]:
        outs_th = outs >= th
        # append onto result dictionary to be returned to function that called `accuracy()`
        res[th] = torch.tensor(torch.sum(outs_th == labels).item() / len(outs))
    return res


class ModelBase(nn.Module):
#     training step
    def train_step(self, batch):
        xb, labels = batch
        outs = self(xb)
        loss = F.binary_cross_entropy(outs, labels)
        return loss
#     validation step
    def val_step(self, batch):
        xb, labels = batch
        outs = self(xb)
        loss = F.binary_cross_entropy(outs, labels)
        acc = accuracy(outs, labels)
        return {'loss': loss.detach(), 'acc': acc}
#     validation epoch (avg accuracies and losses)
    def val_epoch_end(self, outputs):
        batch_loss = [x['loss'] for x in outputs]
        avg_loss = torch.stack(batch_loss).mean()
        batch_acc50 = [x['acc'][.50] for x in outputs]
        batch_acc75 = [x['acc'][.75] for x in outputs]
        batch_acc80 = [x['acc'][.80] for x in outputs]
        avg_acc50 = torch.stack(batch_acc50).mean()
        avg_acc75 = torch.stack(batch_acc75).mean()
        avg_acc80 = torch.stack(batch_acc80).mean()
        return {'avg_loss': avg_loss, 'avg_acc': [avg_acc50, avg_acc75, avg_acc80]}
#     print everything important
    def epoch_end(self, epoch, avgs, test=False):
        s = 'test' if test else 'val'
        print(f'EPOCH {epoch + 1:<10} | {s}_loss:{avgs["avg_loss"]:.3f}, {s}_acc (threshold): (.50){avgs["avg_acc"][0]:.3f}, (.75){avgs["avg_acc"][1]:.3f}, (.80){avgs["avg_acc"][2]:.3f}')


@torch.no_grad()
def evaluate(model, val_dl):
    # eval mode
    model.eval()
    outputs = [model.val_step(batch) for batch in val_dl]
    return model.val_epoch_end(outputs)


def fit(epochs, lr, model, train_dl, val_dl, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    # define optimizer
    optimizer = opt_func(model.parameters(), lr)
    # for each epoch...
    for epoch in range(epochs):
        # training mode
        model.train()
        # (training) for each batch in train_dl...
        for batch in tqdm(train_dl):
            # pass thru model
            loss = model.train_step(batch)
            # perform gradient descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation
        res = evaluate(model, val_dl)
        # print everything useful
        model.epoch_end(epoch, res, test=False)
        # append to history
        history.append(res)
    return history


class VGG16_PT(ModelBase):
    def __init__(self):
        super().__init__()
#         pretrained VGG16 model w/ batch norm
        self.network = torchvision.models.vgg16_bn(pretrained=True)
#         change first layer to accept only 1 dimension of color (b/w)
        self.network.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         change last fc layer to output one value
        self.network.classifier[6] = nn.Linear(4096, 1, bias=True)
        
    def forward(self, xb):
        out = self.network(xb)
        out = F.sigmoid(out)
        return out

# MODEL <<<

# Dataset >>>

class Prudhoe_DS(Dataset):
    def __init__(self, h5_pth, transforms=None):
        self.transforms = transforms
        self.h5_pth = h5_pth
        self.masks = []
        self.labels = []
        
        with h5py.File(self.h5_pth, 'r') as F:
            pos = F['POS']
            neg = F['NEG']
            pos_len = pos.attrs['shape'][0]
            neg_len = neg.attrs['shape'][0]
            
            self.masks = np.vstack([pos, neg])
            self.labels = np.concatenate([np.ones(pos_len),
                                          np.zeros(neg_len)])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mask = self.masks[idx]
        label = self.labels[idx]
        
        return mask, label

# Dataset <<<

SITES = ['prudhoe_12', 'prudhoe_15', 'prudhoe_22']
PTH = '/scratch/richardso21/20-21_BGSUB/'

Path(os.path.join(PTH, 'mask_classifier_M')).mkdir(exist_ok=True)
batch_size = 64

for site in SITES:
    h5F = os.path.join(PTH, 'FgSegNet_O', f'{site}.h5')
    
    ds = Prudhoe_DS(h5F)
    ds_size = len(ds)

    train_size = int(ds_size * .70)
    val_size = (ds_size - train_size) // 2
    test_size = ds_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size*2, pin_memory=True)

    
