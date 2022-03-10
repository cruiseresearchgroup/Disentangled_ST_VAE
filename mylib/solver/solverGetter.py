from .factorVAESolver import FactorVAESolver
from .betaVAESolver import BetaVAESolver

def get_solver(name, cfg, temp_log_dir, temp_ckpt_dir):
    if name == 'factorVAE':
        solver = FactorVAESolver(cfg, temp_log_dir, temp_ckpt_dir)

    elif name == 'betaVAE':
        solver = BetaVAESolver(cfg, temp_log_dir, temp_ckpt_dir)

    return solver

