import os
import json 
import argparse
import copy

import logging
import logging.handlers
from configparser import ConfigParser

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from mylib.solver.solverGetter import get_solver
from mylib.utils import str2bool, mkdirs, EarlyStopping
from mylib.eval.applyingMetrics import apply_metrics, metric_to_csv
from mylib.solver.utils import RMSELoss
from mylib.dataset.datasetGetter import *
from mylib.model.module.baseModel import BaseRegressor
import loadSolver

def _train(net, rep_func, train_loader, optimizer, criterion, length, device, mode):
    net.train()   
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        if mode != 'stgcn':
            x = x.to(device)
            mu, logvar = rep_func.encode_(x)
            z = rep_func.reparameterize(mu, logvar).squeeze()
        else:
            x, adj_mx = x
            adj_mx = Variable(adj_mx[0].squeeze()).to(device)
            x = x.to(device)
            mu, logvar = rep_func.encode_((adj_mx, x))
            z = rep_func.reparameterize(mu, logvar)
            z = z.squeeze()
            
        pred = net(z).double()
        
        y = y.to(device).double()
        # Loss measures generator's ability to fool the discriminator
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return (total_loss / length) ** 0.5
        

def _evalate(net, rep_func, test_loader, criterion, length, device, mode):
    net.eval()
    total_loss = 0
    with torch.no_grad():
        
        for i, (x, y) in enumerate(test_loader):
            if mode != 'stgcn':
                x = x.to(device)
                mu, logvar = rep_func.encode_(x)
                z = rep_func.reparameterize(mu, logvar).squeeze()
            else:
                x, adj_mx = x
                adj_mx = Variable(adj_mx[0].squeeze()).to(device)
                x = x.to(device)
                mu, logvar = rep_func.encode_((adj_mx, x))
                z = rep_func.reparameterize(mu, logvar)
                z = z.squeeze()
                
            pred = net(z).double()
            
            y = y.to(device).double()
            # Loss measures generator's ability to fool the discriminator
            loss = criterion(pred, y)
            total_loss += loss.item()
            
    return (total_loss / length) ** 0.5

def _portion_run(rep_func, total_length, portions, Xs, ys, lr, regressor_params, model_path,
                dset, max_epoch, device,results_dict, model_name, mode, adj=None):
    portion_length = [int(total_length*p) for p in portions]
    train_X, valid_X, test_X = Xs
    train_y, valid_y, test_y = ys

    criterion = torch.nn.MSELoss()
    if mode == 'stgcn' and adj is not None:
        valid_dataset = dset(valid_X, valid_y, adj, y_mode='portion_test')
        test_dataset = dset(test_X, test_y, adj, y_mode='portion_test')
    else:
        valid_dataset = dset(valid_X, valid_y, y_mode='portion_test')
        test_dataset = dset(test_X, test_y, y_mode='portion_test')

    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    for po, po_len in zip(portions, portion_length):
        # print("Start training for portion: {}% with lr:{}".format(int(po*100), lr))
        temp_model_path = model_path + 'checkpoint_{}%_{}.pt'.format(int(po*100), lr)

        temp_x = train_X[:po_len, :]
        temp_y = train_y[:po_len, :]

        if mode == 'stgcn' and adj is not None:
            temp_dataset = dset(temp_x, temp_y, adj, y_mode='portion_test')
        else:
            temp_dataset = dset(temp_x, temp_y, y_mode='portion_test')
        
        temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=True)

        model_reg = BaseRegressor(regressor_params).to(device)
        optim_reg = optim.Adam(model_reg.parameters(), lr=lr)

        early = EarlyStopping(patience=5, verbose=False, delta=0.001, path=temp_model_path)

        for epoch in range(max_epoch):
            if early.early_stop:
                break
                
            train_loss = _train(model_reg, rep_func, temp_loader, optim_reg, criterion, po_len, device, mode)
            valid_loss = _evalate(model_reg, rep_func, valid_loader, criterion, len(valid_dataset), device, mode) 
            early(valid_loss, model_reg)

        model_reg.load_state_dict(torch.load(early.path))
        valid_loss = _evalate(model_reg, rep_func, valid_loader, criterion, len(valid_dataset), device, mode)
        test_loss = _evalate(model_reg, rep_func, test_loader, criterion, len(test_dataset), device, mode)

        print("[Epoch %d/%d] [Train loss: %f] [Valid loss: %f]"
                % (epoch, max_epoch, train_loss, valid_loss)
            )
        
        results_dict['train_portion'].append(int(po*100))
        results_dict['valid_loss'].append(valid_loss)
        results_dict['test_loss'].append(test_loss) 
        results_dict['lr'].append(lr)  
        results_dict['model'].append(model_name)
        
        del temp_x, temp_y, temp_dataset, temp_loader
        torch.cuda.empty_cache()
        print('finish portion test, model: {}, portion:{}, lr:{}'.format(model_name, po, lr))

def eval_portion(config_name):
    cfg = ConfigParser()
    assert os.path.exists('./config/{}.ini'.format(config_name)), "wrong path for the config file"
    cfg.read('./config/{}.ini'.format(config_name))

    assert 'portion_test' in cfg.sections(), "Does not find section: portion_test in the config file"
    assert 'portion_dir' in cfg.options('environment'), 'Does not find: portion_dir in section: environment'

    portion_reg_params = json.loads(cfg['portion_test']['portion_reg_params'])

    dset_class_name = cfg['dataset']['dset_class_name']
    dset_dir = cfg['dataset']['dset_dir']
    dset_name = cfg['dataset']['dset_name']
    mode = json.loads(cfg['model_general']['mode'])
    
    dset = get_eval_dataset(dset_class_name)
    # print(dset)
    data = np.load(dset_dir + dset_name + '.npz')
    # train_dset = dset(data['X'], data['y'], y_mode='portion_test')
    X = data['X']
    if mode[0] == 'stgcn':
        y = data['y_flat']
        adj = data['adj_mx']
    else:   
        y = data['y']
        adj = None


    train_portion = cfg['portion_test'].getfloat('train_portion')
    valid_portion = cfg['portion_test'].getfloat('valid_portion')
    train_portion_list = json.loads(cfg['portion_test']['train_portion_list'])

    data_length = X.shape[0]
    full_train_length = int(data_length * train_portion)
    valid_length = int(data_length * valid_portion)
    # portion_length = [int(full_train_length*p) for p in train_portion_list]

    train_X = X[:full_train_length, :]
    train_y = y[:full_train_length, :]
    valid_X = X[full_train_length:valid_length, :]
    valid_y = y[full_train_length:valid_length, :]
    test_X = X[valid_length:, :]
    test_y = y[valid_length:, :]
    Xs = (train_X, valid_X, test_X)
    ys = (train_y, valid_y, test_y)
    
    portion_dir = './' + cfg.get('environment', 'portion_dir')
    model_path = portion_dir + 'checkpoint/'
    csv_path = portion_dir + 'portion_results_{}.csv'.format(dset_class_name)
    mkdirs(portion_dir)
    mkdirs(model_path)

    max_epoch = cfg['portion_test'].getint('max_epoch')
    lr_list = json.loads(cfg['portion_test']['lr_list'])
    portion_reg_params = json.loads(cfg['portion_test']['portion_reg_params'])

    results_dict = {
        'train_portion': [],
        'valid_loss': [],
        'test_loss': [],
        'lr': [],
        'model': [],
    } 

    solvers = loadSolver.load_model(config_name)
    for solver in solvers:
        representation_func = solver.VAE
        representation_func.eval()

        for lr in lr_list:
            _portion_run(representation_func, 
                        full_train_length, 
                        train_portion_list, 
                        Xs, ys, lr, 
                        portion_reg_params, model_path, 
                        dset=dset, 
                        max_epoch=max_epoch, 
                        device=solver.device,
                        results_dict=results_dict, 
                        model_name=solver.model_name,
                        mode=mode[0],
                        adj=adj)
    
    temp_df = pd.DataFrame.from_dict(results_dict)
    temp_df.to_csv(csv_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentangle-VAE')
    parser.add_argument('--config_name', default=None, type=str, help='the config file path')
    args = parser.parse_args()

    eval_portion(args.config_name)






        