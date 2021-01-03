#!/usr/bin/env python
# coding: utf-8

import os, torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, Dataset
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# select gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# yield accuracy of outputs vs. labels
def accuracy(outs, labels):
    res = {}
    # check accuracy for 3 different thresholds
    for th in [.50, .75, .80]:
        outs_th = outs >= th
        # append onto result dict
        res[th] = torch.tensor(torch.sum(outs_th == labels).item() / len(outs))
    return res

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
#         for i in range(10):
#             print(outputs[i])
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

class CNNModel(ModelBase):
    def __init__(self):
        super().__init__()                                       # 1 x 1024 x 720
#         custom-defined model w/o pretrained weights
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

