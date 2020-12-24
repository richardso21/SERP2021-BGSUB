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
    # train step
    def train_step(self, batch):
        xb, labels = batch
        labels = labels.view(-1, 1)
        outs = self(xb)
        loss = F.binary_cross_entropy(outs, labels)
        return loss
    # validation step