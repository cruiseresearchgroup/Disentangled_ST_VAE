import os
import math
import csv
import torch
import json
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from .effectMetric import effect_metric
from mylib.dataset.datasetGetter import *       

def get_metric(metric_name):
    if metric_name == 'effect':
        return effect_metric

def prepare_kwargs(metric_name, model_solver, cfg):
    assert metric_name in cfg.sections(), "Don't got metric detail for metric: {}".format(metric_name)
    
    dset_class_name = cfg[metric_name]['dset_class_name']
    dset = get_dataset(dset_class_name, cfg)
    num_train_x = len(dset)
    batch_size = model_solver.batch_size

    mode = json.loads(cfg['model_general']['mode'])
    
    x, y = dset[np.random.choice(num_train_x, batch_size)]
    if mode[0] != 'stgcn':
        x = x.to(model_solver.device)
        x_recon, mu, logvar, z = model_solver.VAE(x)
    else:
        x, adj_mx = x
        adj_mx = Variable(adj_mx).to(model_solver.device)
        x = x.to(model_solver.device)
        mu, logvar = model_solver.VAE.encode_((adj_mx, x))
        z = model_solver.VAE.reparameterize(mu, logvar)
        x_recon = model_solver.VAE.decode_((adj_mx, z))
        z = z.squeeze()

    x_recon = F.sigmoid(x_recon)
    
    if metric_name == 'effect':
        kw = {
            'y': y,
            'regressor_params': json.loads(cfg[metric_name]['regressor_params']),
            'lr_reg': cfg[metric_name].getfloat('lr_reg'),
            'M_N': batch_size / num_train_x,
            'epoches_reg': cfg[metric_name].getint('epoches_reg'),
            'mode': mode
        }
    return x, x_recon, mu, logvar, z, kw


def metric_to_csv(path, loss_dict):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=loss_dict.keys()) 
            writer.writeheader() 
            writer.writerow(loss_dict)  
    else:
        with open(path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(loss_dict.values())
        
   
def apply_metrics(eval_metrics, model_solver, cfg):
    total_loss_dict = {}
    for metric_name in eval_metrics:
        loss_dict = {}
        model_solver.logger.info('{}: start evaluate metric, name: {}'.format(model_solver.model_name, metric_name))
        metric = get_metric(metric_name)
        x, recon, mu, log_var, z, kw = prepare_kwargs(metric_name, model_solver, cfg)

        loss_dict = metric(x, recon, mu, log_var, z, **kw)
        total_loss_dict[metric_name] = loss_dict
        
        for k, v in loss_dict.items():
            model_solver.logger.info('{}: {}: {}'.format(model_solver.model_name, k ,v))

        model_solver.logger.info('{}: finished evaluate metric, name: {}'.format(model_solver.model_name, metric_name))

    return total_loss_dict







