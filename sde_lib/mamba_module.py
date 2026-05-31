import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from .sde_utils import associative_scan 

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.act(x_branch)

        x_dbl = self.x_proj(x_branch)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        
        A = -torch.exp(self.A_log)
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        dB = torch.einsum('bld,bln->bldn', dt, B)
        
        u = x_branch
        dB_u = dB * u.unsqueeze(-1)

        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(seq_len):
            h = dA[:, i] * h + dB_u[:, i]
            y = torch.einsum('bdn,bln->bd', h, C[:, i].unsqueeze(1))
            ys.append(y)
        y = torch.stack(ys, dim=1)
        
        y = y + u * self.D
        y = y * self.act(z_branch)
        return self.out_proj(y)

class MambaEncoder(nn.Module):
    def __init__(self, args, width, obs_embedder):
        super().__init__()
        self.obs_emb = obs_embedder
        self.n_layers = args.n_layer
        self.layers = nn.ModuleList([
            MambaBlock(d_model=width, d_state=16, expand=2)
            for _ in range(self.n_layers)
        ])
        self.norm_f = nn.LayerNorm(width)

    def forward(self, obs, event_time, obs_mask=None, event_mask=None):

        if obs_mask is None:
            obs_mask = torch.ones_like(obs)
            
        input_obs = torch.cat([obs, obs_mask], dim=-1)       
        x = self.obs_emb(input_obs)
        y_observed = x.clone().detach()

        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        return x, y_observed
