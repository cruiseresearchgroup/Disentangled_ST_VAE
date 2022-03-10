import os
import json 
import argparse
import logging
import logging.handlers
from configparser import ConfigParser
import copy

import numpy as np
import torch

from mylib.solver.solverGetter import get_solver
from mylib.utils import str2bool, mkdirs
from mylib.eval.applyingMetrics import apply_metrics, metric_to_csv

def single_run(init_seed, exp_models, logger, cfg, gamma=None, temp_log_dir=None, temp_ckpt_dir=None):
    for model in exp_models:
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed(init_seed)
        np.random.seed(init_seed)

        model_solver = get_solver(model, cfg, temp_log_dir, temp_ckpt_dir)
        if gamma:
            model_solver.gamma = gamma
            if model_solver.log_flag:
                model_solver.logger.debug('factorVAE: set gamma to {}'.format(gamma))
        ckpt_load = cfg[model].getboolean('ckpt_load')
        if not ckpt_load:
            model_solver.train()
            logger.info('{}: training stage finished.'.format(model))

        if 'eval' in cfg.options("model_general") and cfg['model_general'].getboolean('eval'):
            eval_metrics = json.loads(cfg['model_general']['eval_metrics'])
            total_loss_dict = apply_metrics(eval_metrics, model_solver, cfg)
            for metric_name, loss_dict in total_loss_dict.items():
                temp_dict = copy.deepcopy(loss_dict)
                temp_dict['seed'] = init_seed
                temp_dict['gamma'] = gamma
                csv_path = './' + cfg.get('environment', 'log_dir') + '{}_{}.csv'.format(model, metric_name)
                metric_to_csv(csv_path, temp_dict)

        logger.removeHandler(model_solver.model_handler)

def main(args):

    cfg = ConfigParser()
    assert os.path.exists('./config/{}.ini'.format(args.config_name)), "wrong path for the config file"
    cfg.read('./config/{}.ini'.format(args.config_name))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    result_dir = './' + cfg.get('environment', 'ckpt_dir')
    log_dir = './' + cfg.get('environment', 'log_dir')

    mkdirs(result_dir)
    mkdirs(log_dir)

    logger = logging.getLogger('rootlogger')
    logger.setLevel(logging.DEBUG)
    
    root_handler = logging.FileHandler(log_dir+'root_log.log')
    root_handler.setLevel(logging.INFO)
    root_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(root_handler)
    logger.addHandler(logging.StreamHandler())

    # print('Exp starts.')
    logger.info('Exp starts.')

    exp_models = json.loads(cfg['model_general']['model_applied'])
    if 'gamma_train' in cfg.options("model_general") and cfg['model_general'].getboolean('gamma_train'):
        gammas = json.loads(cfg['model_general']['gammas'])
    else:
        gammas = []

    if 'seed_train' in cfg.options("model_general") and cfg['model_general'].getboolean('seed_train'):
        seeds = json.loads(cfg['model_general']['seeds'])
    else:
        init_seed = cfg.getint('exp', 'random_seed')
        seeds = [init_seed]
    
    if len(seeds) == 1 and not gammas:
        single_run(seeds[0], exp_models, logger, cfg)
    
    else:
        for seed in seeds:
            for gamma in gammas:
                temp_log_dir = './' + cfg['environment']['log_dir'] + '{}_{}/'.format(seed, gamma)
                temp_ckpt_dir = './' + cfg['environment']['ckpt_dir'] + '{}_{}/'.format(seed, gamma)

                mkdirs(temp_log_dir)
                mkdirs(temp_ckpt_dir)
                single_run(seed, exp_models, logger, cfg, gamma, temp_log_dir, temp_ckpt_dir) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disentangle-VAE')
    parser.add_argument('--config_name', default=None, type=str, help='the config file path')
    args = parser.parse_args()

    main(args)


