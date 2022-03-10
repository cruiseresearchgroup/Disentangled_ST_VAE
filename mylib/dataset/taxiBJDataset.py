import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from mylib.dataset import utils

class TaxiBJDataset(Dataset):
    """TaxiBJ dataset."""

    def __init__(self, X, y, x_norm=False, y_mode='eval'):
        self.X = X
        self.x_norm = x_norm
        self.x_scaler = None
    
        
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        if y_mode == 'eval':
            self.y = self.y[:, :1024]

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        temp_x = self.X[idx]
        temp_y = self.y[idx]
        
        return temp_x, temp_y

class TaxiBJFactorDataset(Dataset):
    """
        The ground-truth factors of variation are:
        0 - average amplitude (6 different values: 0-5)
        1 - hotspot timestep index (21 different values: 0-20)
        2 - the postion of the hotspot (4 different values:0-3)
    """

    def __init__(self, X, y, features):
        self.X = X
        self.x_scaler = None
        self.factor_sizes = [6, 21, 4]
        self.latent_factor_indices = [0, 1, 2]
        self.num_total_factors = features.shape[1]
        self.index = util.StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)
        
        self.data_shape[21, 2, 32, 32]
        
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return self.data_shape
    
    def preprocess_(self):
        if self.x_norm:
            self.x_scaler = MinMaxScaler(feature_range=(0, 1))
#             self.x_scaler = StandardScaler()
            self.X = self.x_scaler.fit_transform(self.X)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        temp_x = self.X[idx]
        temp_y = self.y[idx]
        return temp_x, temp_y

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return self.images[indices].astype(np.float32)

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]