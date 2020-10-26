#!/usr/bin/env python
# coding: utf-8

# In[43]:


import h5py, os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ## Outline
#  - Generate a specific model for each scene
#      - Each location: 1600 images w/o presence, 800 imags w/ presence
#  - Create Dataloader
#      - 80-20 split for training/testing (validation?)
#      - Each img (input size): 1024 x 720 px
#  - Model Architecture
#      - Modified VGG16?
#      - Custom CNN architecture?
#      - Concatenate pixel values and put in logistic regression?

# ---

# ## Model Base/Helpers/Train & Test functions

# In[39]:


def accuracy(outs, labels):
    res = {}
    # check accuracy for 3 different thresholds
    for th in [.50, .75, .80]:
        outs_th = outs >= th
        # append onto result dictionary to be returned to function that called `accuracy()`
        res[th] = torch.tensor(torch.sum(outs_th == labels).item() / len(outs))
    return res


# In[41]:


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


# In[44]:


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


# ---

# ## Models

# In[52]:


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
        out = network(xb)
        out = F.sigmoid(out)
        return out


# In[54]:


class VGG11_PT(ModelBase):
    def __init__(self):
        super().__init__()
#         pretrained VGG16 model w/ batch norm
        self.network = torchvision.models.vgg11_bn(pretrained=True)
#         change first layer to accept only 1 dimension of color (b/w)
        self.network.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         change last fc layer to output one value
        self.network.classifier[6] = nn.Linear(4096, 1, bias=True)
        
    def forward(self, xb):
        out = network(xb)
        out = F.sigmoid(out)
        return out


# In[ ]:


# class Custom_NPT(ModelBase):
#     def __init__(self):
#         super().__init__()                                       # 1 x 1024 x 720
# #         custom-defined model w/o pretrained weights
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # 8 x 1024 x 720
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 16 x 1024 x 720
#         self.pool1 = nn.MaxPool2d(2)                             # 16 x 512 x 360
#         self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 32 x 512 x 360
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32 x 512 x 360
#         self.pool2 = nn.MaxPool2d(2)                             # 32 x 256 x 180
#         self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 16 x 256 x 180
#         self.pool3 = nn.MaxPool2d(2)                             # 8 x 256 x 180
#         self.fc1 = nn.Linear(8*256*180, 4096)
#         self.fc2 = nn.Linear(4096, 512)
#         self.fc3 = nn.Linear(512, 1)
        
#     def forward(self, xb):
#         out = F.relu(self.conv1(xb))
#         out = F.relu(self.conv2(out))
#         out = self.pool1(out)
#         out = F.relu(self.conv3(out))
#         out = F.relu(self.conv4(out))
#         out = self.pool2(out)
#         out = F.relu(self.conv5(out))
#         out = self.pool3(out)
        
#         out = torch.flatten(out, 1)
#         out = F.relu(self.fc1(out))


# In[65]:


class VGG16_NPT(ModelBase):
    def __init__(self):
        super().__init__()
#         pretrained VGG16 model w/ batch norm
        self.network = torchvision.models.vgg16_bn(pretrained=False)
#         change first layer to accept only 1 dimension of color (b/w)
        self.network.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         change last fc layer to output one value
        self.network.classifier[6] = nn.Linear(4096, 1, bias=True)
        
    def forward(self, xb):
        out = network(xb)
        out = F.sigmoid(out)
        return out


# ---

# In[ ]:




