#!/usr/bin/env python
# coding: utf-8

import os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# set seed for reproducability
torch.manual_seed(42)
# environment variable for specifying GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# training batch size
batch_size = 4
# specify main site
site = 'prudhoe_12'


def accuracy(outs, labels):
    # check accuracy for threshold @ 0.7
    outs_th = outs >= 0.7
    # return ratio of outputs >= 0.7 to total evaluated outputs
    return torch.tensor(torch.sum(outs_th == labels).item() / len(outs))

class ModelBase(nn.Module):
#     training step
    def train_step(self, batch):
        xb, labels = batch
        labels = labels.view(-1, 1)
        outs = self(xb)
        loss = F.binary_cross_entropy(outs, labels)
        return loss

#     validation step
    def val_step(self, batch):
        xb, labels = batch
        labels = labels.view(-1, 1)
        outs = self(xb)
        loss = F.binary_cross_entropy(outs, labels)
        acc = accuracy(outs, labels)
        return {'loss': loss.detach(), 'acc': acc}

#     validation epoch (avg accuracies and losses)
    def val_epoch_end(self, outputs):
        batch_loss = [x['loss'] for x in outputs]
        avg_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['acc'] for x in outputs]
        avg_acc = torch.stack(batch_acc).mean()
        return {'avg_loss': avg_loss, 'avg_acc': avg_acc}

#     print everything important
    def epoch_end(self, epoch, avgs, test=False):
        s = 'test' if test else 'val'
        print(f'EPOCH {epoch + 1:<10} | {s}_loss:{avgs["avg_loss"]:.3f}, {s}_acc (0.7 thr): {avgs["avg_acc"]:.3f}')


@torch.no_grad()
def evaluate(model, val_dl):
    # evaluate the model's performance using `val_step` method
    # eval mode
    model.eval()
    outputs = [model.val_step(batch) for batch in val_dl]
    return model.val_epoch_end(outputs)

@torch.no_grad()
def test_evaluate(model, test_dl):
    # modification of `evaluate` that additionally returns raw output as np arrays
    model.eval()
    outputs = np.array([])
    test_labels = np.array([])
    for batch in test_dl:
        xb, labels = batch
        outs = model(xb).data.view(-1)
        outputs = np.append(outputs, outs.cpu().numpy())
        test_labels = np.append(test_labels, labels.cpu().numpy())
    return outputs, test_labels

def fit(epochs, lr, model, train_dl, val_dl, save_pth, opt_func=torch.optim.Adam):
    # main function to train/`fit` the model
    torch.cuda.empty_cache()
    history = []
    top = 0
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
        # if val_acc is higher than before, save model
        if res["avg_acc"] > top:
            print('Saving model...')
            torch.save(model.save_dict(), save_pth)
            top = res["avg_acc"]
        # append to history
        history.append(res)
    return history


class Custom_NPT(ModelBase):
    # custom-defined model w/o pretrained weights
    # no use of batch norm
    def __init__(self):
        super().__init__()                                       # 1 x 1024 x 720
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # 8 x 1024 x 720
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 16 x 1024 x 720
        self.pool1 = nn.MaxPool2d(2)                             # 16 x 512 x 360
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1) # 16 x 512 x 360
        self.pool2 = nn.MaxPool2d(2)                             # 16 x 256 x 180
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, padding=1)  # 8 x 256 x 180
        self.pool3 = nn.MaxPool2d(2)                             # 8 x 128 x 90
        self.fc1 = nn.Linear(8*128*90, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 1)
        
    def forward(self, xb):
        out = F.relu(self.conv1(xb))
        out = F.relu(self.conv2(out))
        out = self.pool1(out)
        out = F.relu(self.conv3(out))
        out = self.pool2(out)
        out = F.relu(self.conv5(out))
        out = self.pool3(out)

        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out


class Prudhoe_DS(Dataset):
    # custom-defined dataset class for specific use case
    def __init__(self, npy_pth, transforms=None):
        self.transforms = transforms
        self.npy_pth = npy_pth
        self.masks_loc = []
        self.labels = []
        
        for typ in ['pos', 'neg']:
            for fn in os.listdir(os.path.join(self.npy_pth, typ)):
                
                full_fn = os.path.join(self.npy_pth, typ, fn)
                self.masks_loc.append(full_fn)
                
                label = 1 if typ == 'pos' else 0
                self.labels.append(label)
        
        self.masks_loc = np.array(self.masks_loc)
        self.labels = np.array(self.labels).astype('float32')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mask = np.expand_dims(np.load(self.masks_loc[idx]), axis=0).astype('float32')
        label = self.labels[idx]
        
        if self.transforms:
            mask = self.transforms()(mask)
        
        return mask, label


# specify main path and site
PTH = '/scratch/richardso21/20-21_BGSUB/'

# clarify other needed paths
npy_pth = os.path.join(PTH, 'FgSegNet_O_neg', site)
save_pth = os.path.join(PTH, 'FgSegNet_clf', site)

# initiate dataset and get size
ds = Prudhoe_DS(npy_pth)
ds_size = len(ds)

# split dataset into train, val, and test
train_size = int(ds_size * .70)
val_size = (ds_size - train_size) // 2
test_size = ds_size - train_size - val_size
train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])

# turn datasets into dataloaders
train_loader = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size*2, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size*2, pin_memory=True, num_workers=4)


# get GPU device
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# transfer all loaders and initiated model onto GPU
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)
model = to_device(Custom_NPT(), device)

# begin training!
history = [evaluate(model, val_loader)]
print(f'Commence training on {site}!')
history += fit(20, 1e-4, model, train_loader, val_loader, save_pth + '.pth')
history += fit(10, 1e-6, model, train_loader, val_loader, save_pth + '.pth')

# evaluate against test_loader
evl = evaluate(model, test_loader)
# getting raw outputs and relative ground truths
outputs, truths = test_evaluate(model, test_loader)
print('FINAL EVALUATION:' + evl)

# save everything
print('Saving history, raw test output, and test ground truths...')
np.save(save_pth + '_history.npy', np.array(history))
np.save(save_pth + '_outputsRaw.npy', outputs)
np.save(save_pth + '_gTruths.npy', truths)
print(f'Completed training + saving on {site}.')
