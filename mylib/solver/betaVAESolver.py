import os
import logging
import json

import visdom
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import *
from ..utils import *
from .baseSolver import BaseSolver, ModelFilter

from ..model.module.baseModel import BaseDiscriminator
from ..model.betaVAE import BetaVAE
from ..dataset.datasetGetter import *

class BetaVAESolver(BaseSolver):
    # The super class for all solver class, describe all essential functions.
    def __init__(self, cfg, log_dir=None, ckpt_dir=None):
        super(BetaVAESolver, self).__init__(cfg)
        self.model_name = 'betaVAE'

        # Create logging.handler
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = './' + cfg['environment']['log_dir']

        self.log_flag = cfg['environment'].getboolean('log_flag')
        if self.log_flag:
            self.logger = logging.getLogger('rootlogger')
            self.model_handler = logging.FileHandler(self.log_dir+'{}_train_log.log'.format(self.model_name))
            self.model_handler.setLevel(logging.DEBUG)
            self.model_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.log_filter = ModelFilter(self.model_name)
            self.model_handler.addFilter(self.log_filter)
            self.logger.addHandler(self.model_handler)

        # Checkpoint
        if ckpt_dir:
            self.ckpt_dir = ckpt_dir + '{}/'.format(self.model_name)
        else:
            self.ckpt_dir = os.path.join(cfg['environment']['prefix'], cfg['environment']['ckpt_dir'], '{}/'.format(self.model_name))
        self.ckpt_save_iter = cfg['environment'].getint('ckpt_save_iter')
        mkdirs(self.ckpt_dir)
        ckpt_load = cfg[self.model_name].getboolean('ckpt_load')

        if not ckpt_load and self.log_flag:
            self.logger.info('{}: Solver initialising.'.format(self.model_name))

        # Get Data loader
        self.data_loader = get_loader(self.model_name, cfg)
        self.batch_num = len(self.data_loader.dataset) // self.batch_size

        # Networks parameter readin
        self.encoder_layer_dims = json.loads(cfg[self.model_name]['encoder_layer_dims'])
        self.decoder_layer_dims = json.loads(cfg[self.model_name]['decoder_layer_dims'])

        self.objective = cfg[self.model_name]['objective']
        self.beta = cfg[self.model_name].getfloat('beta')
        self.gamma = cfg[self.model_name].getfloat('gamma')
        self.z_dim = cfg[self.model_name].getint('z_dim')

        self.C_max = cfg[self.model_name].getfloat('C_max')
        self.C_stop_iter = cfg[self.model_name].getint('C_stop_iter')

        self.lr_VAE = cfg[self.model_name].getfloat('lr_VAE')
        self.beta1_VAE = cfg[self.model_name].getfloat('beta1_VAE')
        self.beta2_VAE = cfg[self.model_name].getfloat('beta2_VAE')

        self.mode = json.loads(cfg['model_general']['mode'])
        if self.mode[0] == 'stgcn':
            torch.backends.cudnn.enabled = False
        self.input_size = cfg['dataset'].getint('input_size')
        self.channel_num = cfg['dataset'].getint('channel_num')

        self.vae_activation = getattr(nn, cfg[self.model_name]['vae_activation'])
        self.vae_act_param = json.loads(cfg[self.model_name]['vae_act_param'])

        # Network & Optimizer initialise
        self.VAE = BetaVAE(self.encoder_layer_dims, self.decoder_layer_dims, 
                            self.mode, self.vae_activation, self.vae_act_param).to(self.device)
        self.optim = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))
        
        # Check network structure
        # if not ckpt_load and self.log_flag:
        #     vae_summary = generate_summary(self.VAE, self.mode, self.input_size, cfg)
        #     self.logger.debug('{}: model summary-VAE \n'.format(self.model_name) + str(vae_summary))

        self.net = self.VAE

        # Checkpoint loading
        if ckpt_load:
            self.load_checkpoint(verbose=ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(cfg['environment']['prefix'], cfg['environment']['output_dir'], '{}/'.format(self.model_name))
        self.output_save = cfg['environment'].getboolean('output_save')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def train(self): 
        if self.log_flag:
            self.logger.info('{}: training starts.'.format(self.model_name))
        self.net_mode(train_flag=True)

        self.C_max = Variable(torch.FloatTensor([self.C_max]).to(self.device))

        progress_template = 'betaVAE: [{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'

        out = False
        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                if self.mode[0] != 'stgcn':
                    x = Variable(x.to(self.device))
                    x_recon, mu, logvar, z = self.net(x)
                else:
                    x, adj_mx = x
                    adj_mx = Variable(adj_mx[0].squeeze()).to(self.device)
                    x = Variable(x.to(self.device))
                    mu, logvar = self.net.encode_((adj_mx, x))
                    z = self.net.reparameterize(mu, logvar)
                    x_recon = self.net.decode_((adj_mx, z))
                    z = z.squeeze()

                # x = Variable(x.to(self.device))
                # x_recon, mu, logvar, z = self.net(x)
                vae_recon_loss = recon_loss(x, x_recon)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = vae_recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = vae_recon_loss + self.gamma*(total_kld-C).abs()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                result_str = progress_template.format(self.global_iter, vae_recon_loss.item(), 
                                                        total_kld.item(), mean_kld.item())
                
                if self.log_flag:
                    self.logger.debug(result_str)

                if self.global_iter%self.print_iter == 0:
                    self.pbar.write(result_str)

                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.global_iter >= self.max_iter:
                    out = True
                    break
        if self.log_flag:
            self.logger.info('{}: training finished.'.format(self.model_name))
        self.pbar.write("[Training Finished]")
        self.pbar.close()
        self.model_handler.close()

    def train_iter(self, x):
        self.net_mode(train_flag=True)
        
        x = Variable(x.to(self.device))
        self.C_max = Variable(torch.FloatTensor([self.C_max]).to(self.device))
        x_recon, mu, logvar, z = self.net(x)
        vae_recon_loss = recon_loss(x, x_recon)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        if self.objective == 'H':
            beta_vae_loss = vae_recon_loss + self.beta*total_kld
        elif self.objective == 'B':
            C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
            beta_vae_loss = vae_recon_loss + self.gamma*(total_kld-C).abs()

        self.optim.zero_grad()
        beta_vae_loss.backward()
        self.optim.step()

    def net_mode(self, train_flag):
        if not isinstance(train_flag, bool):
            raise ValueError('Only bool type is supported. True|False')

        if train_flag:
            self.net.train()
        else:
            self.net.eval()
    
    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname=-1, verbose=True):
        if ckptname == -1:
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            verbose_text = "betaVAE: => loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter)
        else:
            verbose_text = "betaVAE: => no checkpoint found at '{}'".format(filepath)
