import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class MelbPedDataset(Dataset):
    """MelbPedDataset dataset."""

    def __init__(self, X, y, adj=None, y_mode='eval'):
        self.X = X
        self.x_scaler = None
        
        self.adj = adj
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        if y_mode == 'eval':
            self.y = self.y[:, :35]
    

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        temp_x = self.X[idx]
        temp_y = self.y[idx]

        if self.adj is not None:
            A_hat = torch.tensor(self.adj)
            temp_x = (temp_x, A_hat)
        
        return temp_x, temp_y