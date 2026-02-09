import torch
import torch.nn as nn
import sys
import math
sys.path.append("sde_lib")
from sde_lib.sde import LinearSDE



class SDE_RUL_Adapter(nn.Module):
    def __init__(self, opt):
        super().__init__()

        class SDEArgs: pass
        args = SDEArgs()
        args.dataset = 'custom_rul'
        args.task = 'regression'
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.state_dim = opt.dim_en
        args.num_basis = 15
        args.lamda_1 = 0.01
        args.init_sigma = 0.1
        args.ts = 1.0
        args.out_dim = 1
        
        args.num_features = opt.num_features 
        
        args.n_layer = opt.num_enc_layers
        args.drop_out = opt.drop_out
        args.info_type = "history"
        
        self.seq_len = opt.train_seq_len
        self.sde_model = LinearSDE(args)

        self.regressor = nn.Sequential(
            nn.Linear(opt.dim_en, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.aux_loss = 0

    def forward(self, x, obs_times=None, mask=None, y_target=None):
        batch_size = x.shape[0]
        
        if obs_times is None:
            obs_times = torch.arange(self.seq_len, dtype=torch.float32, device=x.device)
            obs_times = obs_times.unsqueeze(0).repeat(batch_size, 1) / self.seq_len
            
        if mask is None:
            mask = torch.ones_like(x)

        obs_valid = (mask.sum(dim=-1) > 0).float()

        means, loss_sde, health_index = self.sde_model(
            x, obs_times, obs_valid, None, 
            rul_target=y_target
        )
        
        self.aux_loss = loss_sde
        last_state = means[:, -1, :]
        rul_pred_mlp = self.regressor(last_state)
        rul_pred_phy = health_index[:, -1].unsqueeze(-1)
        final_prediction = 0.5 * rul_pred_mlp + 0.5 * rul_pred_phy
        
        return final_prediction
