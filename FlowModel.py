import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random  

def get_timestep_embedding(timesteps, embedding_dim: int):
    timesteps = timesteps.to(dtype=torch.float32)
    assert len(timesteps.shape) == 1 
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    assert emb.shape == torch.Size([timesteps.shape[0], embedding_dim])
    return emb


class DeltaFMCDR(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, dropout=0.1):
        super(DeltaFMCDR, self).__init__()
        
        self.input_dim = input_dim
        self.diff_dim = hidden_dim 

        self.cond_emb_linear = nn.ModuleList(
            [   
                nn.Linear(input_dim,input_dim),
            ]
        ) 


        self.linears = nn.ModuleList([
            nn.Linear(self.input_dim, self.diff_dim),      
            nn.Linear(self.diff_dim, self.diff_dim),      
            nn.Linear(self.diff_dim, input_dim),   
        ])
        

        self.step_emb_linear = nn.Linear(self.diff_dim, input_dim)

        self.al_linear = nn.Linear(input_dim, input_dim, bias=False)

    def forward_velocity(self, x, t, cond_emb):

        t_scaled = t * 1000.0
        t_embedding = get_timestep_embedding(t_scaled, self.diff_dim) 
        t_embedding = self.step_emb_linear(t_embedding)  

        cond_embedding = self.cond_emb_linear[0](cond_emb)      
        
        x_in = x+t_embedding+cond_embedding

        x = x_in 

        x = self.linears[0](x) 
        x = self.linears[1](x) 
        x = self.linears[2](x) 

        return x 

    def get_al_emb(self, emb):
        return self.al_linear(emb)

    def calculate_loss(self, u_src, u_tgt, i_tgt=None, y_true=None, is_task=False):
        B, D = u_src.shape
        device = u_src.device

        x_0 = u_src 
        x_1_pos = u_tgt 
        

        t = torch.rand(B, device=device)
        t_in = t 
        t_reshaped = t.view(B, 1) 

        x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1_pos


        v_pred = self.forward_velocity(x_t, t_in, cond_emb=x_0)
        

        v_gt = x_1_pos - x_0


        loss_mse = F.mse_loss(v_pred, v_gt)


        if is_task:
            train_infer_steps = 2 
            with torch.no_grad():
                x_1_est = self._ode_solve(x_0, steps=train_infer_steps)

            x_1_aligned = self.get_al_emb(x_1_est)
            

            loss_align = F.smooth_l1_loss(x_1_aligned, x_1_pos)
            loss_task = torch.tensor(0.0, device=device)
            if i_tgt is not None and y_true is not None:
                y_pred = torch.sum(x_1_aligned * i_tgt, dim=1) 
                loss_task = F.mse_loss(y_pred, y_true.float().squeeze())

            return loss_align + 0.1 * loss_task 
            
        else:
            return loss_mse 

    @torch.no_grad()
    def map_user(self, u_src, steps=2):

        x_0 = u_src
        x_final = self._ode_solve(x_0, steps) 
        
        return self.get_al_emb(x_final)


    def _ode_solve(self, x_0, steps):

        B = x_0.shape[0]
        device = x_0.device
        dt = 1.0 / steps

        x_curr = x_0  
        
        for i in range(steps): 

            t_val = i / steps
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            
            v = self.forward_velocity(x_curr, t, cond_emb=x_0)

            x_curr = x_curr + v * dt
            
        return x_curr