import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

class BetaVAEDataset(Dataset):
    def __init__(self, X, adj=None, transform=None):
        self.X = X
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.transform = transform
        self.indices = range(len(self))
        self.adj = adj

    def __getitem__(self, index1):
        x1 = self.X[index1]
        if self.transform is not None:
            x1 = self.transform(x1)
        if self.adj is not None:
            A_hat = torch.tensor(self.adj)
            x1 = (x1, A_hat)

        return x1

    def __len__(self):
        return self.X.size(0)