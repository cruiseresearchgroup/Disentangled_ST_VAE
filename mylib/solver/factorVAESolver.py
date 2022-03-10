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
from ..model.factorVAE import FactorVAE
from ..dataset.datasetGetter import *

class FactorVAESolver(BaseSolver):
    # The super class for all solver class, describe all essential functions.
    def __init__(self, cfg, log_dir=None, ckpt_dir=None):
        super(FactorVAESolver, self).__init__(cfg)
        self.model_name = 'factorVAE'

        # Create logging.handler
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = './' + cfg['environment']['log_dir']

        # Checkpoint
        if ckpt_dir:
            self.ckpt_dir = ckpt_dir + 'factorVAE/'
        else:
            self.ckpt_dir = os.path.join(cfg['environment']['prefix'], cfg['environment']['ckpt_dir'], 'factorVAE/')
        self.ckpt_save_iter = cfg['environment'].getint('ckpt_save_iter')
        mkdirs(self.ckpt_dir)
        ckpt_load = cfg[self.model_name].getboolean('ckpt_load')

        self.log_flag = cfg['environment'].getboolean('log_flag')
        if self.log_flag:
            self.logger = logging.getLogger('rootlogger')
            self.model_handler = logging.FileHandler(self.log_dir+'{}_train_log.log'.format(self.model_name))
            self.model_handler.setLevel(logging.DEBUG)
            self.model_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.log_filter = ModelFilter(self.model_name)
            self.model_handler.addFilter(self.log_filter)
            self.logger.addHandler(self.model_handler)

        if not ckpt_load and self.log_flag:
            self.logger.info('{}: Solver initialising.'.format(self.model_name))
        
        # Get Data loader
        self.data_loader = get_loader(self.model_name, cfg)
        self.dset_name = cfg['dataset']['dset_name']

        # exp parameter readin
        self.print_iter = cfg['exp'].getint('print_iter')

        # Networks parameter readin
        self.encoder_layer_dims = json.loads(cfg[self.model_name]['encoder_layer_dims'])
        self.decoder_layer_dims = json.loads(cfg[self.model_name]['decoder_layer_dims'])
        self.disc_layer_dims = json.loads(cfg[self.model_name]['disc_layer_dims'])

        self.gamma = cfg[self.model_name].getfloat('gamma')
        self.z_dim = cfg[self.model_name].getint('z_dim')

        self.lr_VAE = cfg[self.model_name].getfloat('lr_VAE')
        self.beta1_VAE = cfg[self.model_name].getfloat('beta1_VAE')
        self.beta2_VAE = cfg[self.model_name].getfloat('beta2_VAE')

        self.lr_D = cfg[self.model_name].getfloat('lr_D')
        self.beta1_D = cfg[self.model_name].getfloat('beta1_D')
        self.beta2_D = cfg[self.model_name].getfloat('beta2_D')
        self.mode = json.loads(cfg['model_general']['mode'])
        if self.mode[0] == 'stgcn':
            torch.backends.cudnn.enabled = False
        self.input_size = cfg['dataset'].getint('input_size')
        self.channel_num = cfg['dataset'].getint('channel_num')
        self.seq_len = cfg['dataset'].getint('seq_len')

        self.vae_activation = getattr(nn, cfg[self.model_name]['vae_activation'])
        self.vae_act_param = json.loads(cfg[self.model_name]['vae_act_param'])
        self.d_activation = getattr(nn, cfg[self.model_name]['d_activation'])
        self.d_act_param = json.loads(cfg[self.model_name]['d_act_param'])

        # Network & Optimizer initialise
        self.VAE = FactorVAE(self.encoder_layer_dims, self.decoder_layer_dims, 
                            self.mode, self.vae_activation, self.vae_act_param).to(self.device)
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))
        
        self.D = BaseDiscriminator(self.disc_layer_dims, self.d_activation, self.d_act_param).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))
        
        # Check network structure
        # if not ckpt_load and self.log_flag:
        #     vae_summary = generate_summary(self.VAE, self.mode, self.input_size, cfg)
        #     d_summary = generate_summary(self.D, ['linear'], self.z_dim, cfg) 

        #     self.logger.debug('{}: model summary-VAE \n'.format(self.model_name) + str(vae_summary))
        #     self.logger.debug('{}: model summary-D \n'.format(self.model_name) + str(d_summary))

        self.nets = [self.VAE, self.D]

        # Checkpoint loading
        if ckpt_load:
            ckptname = cfg[self.model_name].getint('ckptname')
            self.load_checkpoint(ckptname=ckptname, verbose=ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(cfg['environment']['prefix'], cfg['environment']['output_dir'], 'factorVAE/')
        self.output_save = cfg['environment'].getboolean('output_save')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def train(self):
        if self.log_flag:
            self.logger.info('{}: training starts.'.format(self.model_name))
        self.net_mode(train_flag=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        progress_template = 'factorVAE: [{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'

        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)
                

                if self.mode[0] != 'stgcn':
                    x_true1 = x_true1.to(self.device)
                    x_recon, mu, logvar, z = self.VAE(x_true1)
                else:
                    x_true1, adj_mx = x_true1
                    adj_mx = Variable(adj_mx[0].squeeze()).to(self.device)
                    x_true1 = x_true1.to(self.device)
                    mu, logvar = self.VAE.encode_((adj_mx, x_true1))
                    z = self.VAE.reparameterize(mu, logvar)
                    x_recon = self.VAE.decode_((adj_mx, z))
                    z = z.squeeze()
                    # x_recon, mu, logvar, z = self.VAE((adj_mx, x_true1))
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld, _, _ = kl_divergence(mu, logvar)

                D_z = self.D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                x_true2 = x_true2.to(self.device)
                if self.mode[0] != 'stgcn':
                    z_prime = self.VAE(x_true2, no_dec=True)
                else:
                    mu_prime, logvar_prime = self.VAE.encode_((adj_mx, x_true2))
                    z_prime = self.VAE.reparameterize(mu_prime, logvar_prime).squeeze()
                # z_prime = self.VAE(x_true2, no_dec=True)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()
            
                result_str = progress_template.format(self.global_iter, vae_recon_loss.item(), 
                                                        vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item())
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
        x_true1, x_true2 = x
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        
        self.global_iter += 1

        x_true1 = x_true1.to(self.device)
        x_recon, mu, logvar, z = self.VAE(x_true1)
        vae_recon_loss = recon_loss(x_true1, x_recon)
        vae_kld, _, _ = kl_divergence(mu, logvar)

        D_z = self.D(z)
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        vae_loss = vae_recon_loss + vae_kld + self.gamma*vae_tc_loss

        self.optim_VAE.zero_grad()
        vae_loss.backward(retain_graph=True)
        self.optim_VAE.step()

        x_true2 = x_true2.to(self.device)
        z_prime = self.VAE(x_true2, no_dec=True)
        z_pperm = permute_dims(z_prime).detach()
        D_z_pperm = self.D(z_pperm)
        D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

        self.optim_D.zero_grad()
        D_tc_loss.backward()
        self.optim_D.step()
        
    def net_mode(self, train_flag):
        if not isinstance(train_flag, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train_flag:
                net.train()
            else:
                net.eval()
    
    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
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
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                print('2')
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))
