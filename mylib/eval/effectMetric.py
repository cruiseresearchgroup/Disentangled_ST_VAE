import os
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from mylib.model.module.baseModel import *
from mylib.solver.utils import *

def effect_metric(x, recon, mu, log_var, z, **kwargs):
    loss_dict = elbo_decomposition(x, recon, mu, log_var, z, **kwargs)

    batch_size, latent_dim = z.shape
    y = kwargs['y'].cuda()
    
    regressor_params = kwargs['regressor_params']
    model = BaseRegressor(regressor_params).to(x.device)
    
    optim_reg = optim.Adam(model.parameters(), lr=kwargs['lr_reg'])
    criterion = torch.nn.MSELoss()


    def utility_loss(model, x, y, epoches, optim_reg, criterion):
        for epoch in range(epoches):
            optim_reg.zero_grad()

            pred = model(z)
            utility_loss = torch.sqrt(criterion(y, pred))
            utility_loss.backward(retain_graph=True)
            optim_reg.step()

        with torch.no_grad():
            pred = model(z)
            utility_loss = torch.sqrt(criterion(y, pred))

        return utility_loss.item()

    loss_dict['utility_loss'] = utility_loss(model, x, y, kwargs['epoches_reg'], optim_reg, criterion)
    loss = loss_dict['recons_loss'] + loss_dict['mi_Loss'] + loss_dict['tc_Loss'] + loss_dict['utility_loss']
    loss_dict['loss'] = loss    
    return loss_dict

def log_density_gaussian(z, mu, log_var):
        """
        Computes the log pdf of the Gaussian with parameters mu and log_var at z
        :param x: (Tensor) Point at which Gaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param log_var: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + log_var)
        log_density = norm - 0.5 * ((z - mu) ** 2 * torch.exp(-log_var))
        return log_density

def NLL(params):
        """Analytically computes
            E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
        If mu_2, and sigma_2^2 are not provided, defaults to entropy.
        """
        mu, logsigma = params
        sample_mu, sample_logsigma = mu, logsigma

        normalization = Variable(torch.Tensor([np.log(2 * np.pi)]))
        c = normalization.type_as(sample_mu.data)
        nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) \
            + torch.exp(sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
        return nll.mul(0.5)

def elbo_decomposition(x, recon, mu, log_var, z, **kwargs):
    mu = mu.squeeze()
    log_var = log_var.squeeze()

    recons_loss = F.mse_loss(recon, x, reduction='sum')
    log_q_zx = log_density_gaussian(z, mu, log_var).sum(dim = 1)

    zeros = torch.zeros_like(z)
    log_p_z = log_density_gaussian(z, zeros, zeros).sum(dim = 1)

    batch_size, latent_dim = z.shape
    mat_log_q_z = log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                            mu.view(1, batch_size, latent_dim),
                                            log_var.view(1, batch_size, latent_dim))
    
    # kwargs['M_N'] = batch_size / num_train_imgs
    # Reference
    # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
    dataset_size = (1 / kwargs['M_N']) * batch_size # dataset size
    strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
    importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(x.device)
    importance_weights.view(-1)[::batch_size] = 1 / dataset_size
    importance_weights.view(-1)[1::batch_size] = strat_weight
    importance_weights[batch_size - 2, 0] = strat_weight
    log_importance_weights = importance_weights.log()

    mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)
    log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
    # # log_q_z = NLL((mu, log_var)).mean(1)
    log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

    mi_loss  = (log_q_zx - log_q_z).mean()
    tc_loss = (log_q_z - log_prod_q_z).mean()
    kld_loss = (log_prod_q_z - log_p_z).mean()

    vae_kld, _, _ = kl_divergence(mu, log_var)

    return {'recons_loss': recons_loss.item()/batch_size,
                'kld': kld_loss.item(),
                'tc_Loss': tc_loss.item(),
                'mi_Loss': mi_loss.item(),
                'vae_kld': vae_kld.item(),}

