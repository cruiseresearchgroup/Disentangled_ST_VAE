import os
import json 
import copy
from configparser import ConfigParser

import torch
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np

import mylib.model
from mylib.solver.solverGetter import get_solver

def load_model(config_name):
    cfg = ConfigParser()
    assert os.path.exists('./config/{}.ini'.format(config_name)), "wrong path for the config file"
    cfg.read('./config/{}.ini'.format(config_name))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if 'gamma_train' in cfg.options("model_general") and cfg['model_general'].getboolean('gamma_train'):
        gammas = json.loads(cfg['model_general']['gammas'])
    else:
        gammas = []

    if 'seed_train' in cfg.options("model_general") and cfg['model_general'].getboolean('seed_train'):
        seeds = json.loads(cfg['model_general']['seeds'])
    else:
        init_seed = cfg.getint('exp', 'random_seed')
        seeds = []

    exp_models = json.loads(cfg['model_general']['model_applied'])   
    solvers = []
    for model in exp_models:
        if seeds and gammas:
            init_seed = seeds[0]
            gamma = gammas[0]
            temp_log_dir = './' + cfg['environment']['log_dir'] + '{}_{}/'.format(init_seed, gamma)
            temp_ckpt_dir = './' + cfg['environment']['ckpt_dir'] + '{}_{}/'.format(init_seed, gamma)
        else:
            temp_log_dir = './' + cfg['environment']['log_dir']
            temp_ckpt_dir = './' + cfg['environment']['ckpt_dir']
        
        
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed(init_seed)
        np.random.seed(init_seed)

        model_solver = get_solver(model, cfg, temp_log_dir, temp_ckpt_dir)
        solvers.append(model_solver)
    
    return solvers

    