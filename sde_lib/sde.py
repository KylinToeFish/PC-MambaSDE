import torch
import torch.nn as nn
import geotorch
from typing import Tuple
from .control import Transformer_Encoder, Decoder
from .jax_compat import associative_scan
from .losses import GNLL_

def elup(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)

@torch.jit.script
def binary_operator(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    A_i, Bu_i = q_i
    A_j, Bu_j = q_j
    return A_j * A_i, torch.addcmul(Bu_j, A_j, Bu_i)

def init_orthogonal(m):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight, 1)

class LinearSDE(torch.nn.Module):
    def __init__(self, args):
        super(LinearSDE, self).__init__()
        self.args = args
        self.ld = args.state_dim
        self.nb = args.num_basis
        
        self.E = nn.Linear(self.ld, self.ld, bias=False)
        self.E.apply(init_orthogonal)
        geotorch.orthogonal(self.E, "weight")
        
        self.D = nn.Parameter(torch.randn(self.nb, self.ld))
        
        self.init_mean = torch.nn.Parameter(torch.randn(self.ld))
        self.init_log_var = torch.nn.Parameter(torch.randn(self.ld))

        self.coeff_net = nn.Sequential(nn.Linear(self.ld, self.nb), nn.Softmax(dim=-1))
        
        self.encoder = Transformer_Encoder(args)
        
        self.decoder = Decoder(args) 

        self.B = nn.Linear(self.ld, self.ld, bias=False)
        
        self.base_degradation = nn.Parameter(torch.tensor([-0.5])) 

    def get_matrix(self, alpha, obs_times, sigma=1):
        Identity = torch.ones(alpha.shape[-1], device=alpha.device)
        
        A_basis = - (elup(self.D) + 1e-6)
        A_coeff = self.coeff_net(alpha)
        A_mat = (A_coeff[..., None] * A_basis[None]).sum(1)
    
        phy_drift = torch.zeros_like(A_mat)
    
        phy_drift[:, 0] = -torch.abs(self.base_degradation) 
        
        A_final = A_mat + phy_drift
        
        exp_A_mat_m = torch.exp(A_final * obs_times)
        exp_B_mat_m = (1/A_final) * (exp_A_mat_m - Identity) * alpha

        exp_A_mat_v = torch.exp(2 * A_final * obs_times)
        exp_B_mat_v = 0.5 * sigma**2 * (1/A_final) * (exp_A_mat_v - Identity) + 1e-6

        return torch.cat([exp_A_mat_m, exp_A_mat_v], dim=-1), torch.cat([exp_B_mat_m, exp_B_mat_v], dim=-1)

    def parallel_compute(self, init, E, Z, obs_times):

        alphas = torch.vmap(lambda u: self.B(u))(Z)
        mats_A, mats_B = torch.vmap(lambda a, t: self.get_matrix(a, t))(alphas, obs_times)
        cum_initial, cum_integral = associative_scan(binary_operator, (mats_A, mats_B))
        
        init_mean_var = torch.vmap(lambda cum_init : cum_init * init)(cum_initial)
        init_mean, init_var = torch.vmap(lambda mean_var : torch.chunk(mean_var, chunks=2, dim=1))(init_mean_var)
        xs_mean, xs_var = torch.vmap(lambda mean_var : torch.chunk(mean_var, chunks=2, dim=1))(cum_integral)
        
        means = xs_mean + init_mean
        stds = torch.sqrt(xs_var + init_var + 1e-6)
        
        return means, stds, alphas

    def forward(self, obs, obs_times, obs_valid, mask_obs, n_samples=1, rul_target=None):

        obs_times_ = self.args.ts * obs_times
        
        Z, y_observed = self.encoder(obs, obs_times_, obs_mask=mask_obs, event_mask=obs_valid)
        
        obs_times = obs_times_[..., None]

        E = self.E.weight.data
        init_mean = E.t() @ self.init_mean
        
        init_mean[0] = 1.0 
        
        init_var = E.t() @ (0.1 * (elup(self.init_log_var) + 1e-6).diag_embed()) @ E
        init_var = torch.diag(init_var)
        init_mean_var = torch.cat([init_mean, init_var], dim=0)

        means, stds, alphas = self.parallel_compute(init_mean_var, self.E, Z, obs_times)
        
        health_index = means[..., 0] 
    
        kl_loss = 0.5 * (alphas.pow(2)).mean()
        
        aux_loss = self.args.lamda_1 * kl_loss
        
        bridge_loss = 0
        if rul_target is not None:
            
            final_hi = health_index[:, -1]
            target_hi = rul_target.squeeze()

            bridge_loss = nn.MSELoss()(final_hi, target_hi)

            hi_diff = health_index[:, 1:] - health_index[:, :-1]
            monotonic_loss = torch.relu(hi_diff).mean()
            
            aux_loss += bridge_loss * 10.0 + monotonic_loss * 1.0

        feat_flat = means 
        
        return means, aux_loss, health_index
