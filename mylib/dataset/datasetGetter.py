import os
import random
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from . import factorVAEDataset, betaVAEDataset
from . import taxiBJDataset, melbPedDataset, bikeNYCDataset
from .imageDataset import return_data

def get_solver_dataset(model_name):
    if model_name == 'factorVAE':
        return factorVAEDataset.FactorVAEDataset
    elif model_name == 'betaVAE':
        return betaVAEDataset.BetaVAEDataset

def get_loader(model_name, cfg):
    dset_dir = cfg['dataset']['dset_dir']
    dset_name = cfg['dataset']['dset_name']

    batch_size = cfg['exp'].getint('batch_size')
    num_workers = cfg['exp'].getint('num_workers')

    dset = get_solver_dataset(model_name)

    mode = json.loads(cfg['model_general']['mode'])
    
    if mode[0] == 'linear':
        with open(dset_dir + dset_name + '/train.npy', 'rb') as f:
            data = np.load(f)

        train_dset = dset(data)

    elif mode[0] == 'conv':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'])
    
    elif mode[0] == 'series':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'])

    elif mode[0] == 'spatio_temporal' or "st-resnet":
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'])

    elif mode[0] == 'stgcn':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'], data['adj_mx'])
    

    train_loader = DataLoader(train_dset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader

def get_eval_dataset(dset_class_name):
    if dset_class_name == 'taxiBJ':
        return taxiBJDataset.TaxiBJDataset
    elif dset_class_name == 'melbPed':
        return melbPedDataset.MelbPedDataset
    elif dset_class_name == 'bikeNYC':
        return bikeNYCDataset.BikeNYCDataset

def get_dataset(dset_class_name, cfg):
    dset_dir = cfg['dataset']['dset_dir']
    dset_name = cfg['dataset']['dset_name']

    dset = get_eval_dataset(dset_class_name)

    mode = json.loads(cfg['model_general']['mode'])
    
    if mode[0] == 'series':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'], data['y'])
    
    elif mode[0] == 'conv':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'], data['y'])

    elif mode[0] == 'spatio_temporal' or mode[0] == "st-resnet":
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'], data['y'])
    
    elif mode[0] == 'stgcn':
        eval_flag = cfg['model_general'].getboolean('eval')
        data = np.load(dset_dir + dset_name + '.npz')
        if eval_flag:
            train_dset = dset(data['X'], data['y_flat'], data['adj_mx'])
        else:
            train_dset = dset(data['X'], data['y_flat'], data['adj_mx'], y_mode='down')

    return train_dset

