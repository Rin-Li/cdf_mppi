import numpy as np
import torch
from typing import Optional
from torch import Tensor
from torch.distributions import MultivariateNormal
from dataclasses import dataclass

@dataclass
class MPPIConfig:
    d_action: int # Dof
    goal: Tensor
    horizon: int = 1 
    n_samples: int = 200
    dt: float = 0.01
    beta: float = 1.0 # inverse temperature
    u_clip: float = 2.7 # elementwise symmetric clip, if not using proj_fn


    # distribution smoothing & stability
    alpha_mean: float = 0.7
    alpha_cov: float = 0.7
    cov_reg: float = 1e-2


    # init distribution
    init_mean: Optional[Tensor] = None # [H, d]
    init_cov_diag: float = 1.0


    # runtime
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    
    goal: Tensor

class MPPI:
    def __init__(
    self,
    cfg: MPPIConfig,
    dynamics,
    cost_fn,
    proj_fn: None,
    store_samples: bool = False,
    ) -> None:
        self.cfg = cfg
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.proj_fn = proj_fn
        self.store_samples = store_samples
        
        
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        self.dof, self.H = cfg.d_action, cfg.horizon
        self.n_sample = cfg.n_samples
        self.cons_u = cfg.u_clip
        self.alpha_mean = cfg.alpha_mean
        self.beta = cfg.beta
        self.alpha_cov = cfg.alpha_cov
        self.cov_reg = cfg.cov_reg
        self.beta = cfg.beta 
        self.dt = cfg.dt
        self.goal = cfg.goal
        
        
        # Distribution
        self.mean = (cfg.init_mean.clone().to(self.device) if cfg.init_mean is not None
                   else torch.zeros(self.H, self.dof, device=self.device, dtype=self.dtype))
        
        eye = torch.eye(self.dof, device=self.device, dtype=self.dtype)
        self.cov = torch.stack([cfg.init_cov_diag * eye for _ in range(self.H)], dim=0)
        
        self.u_all_sample = []
        self.cost_all = []
        self.q_all = []
        self.u_exec_all = []
        
        
    
    def sample(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        flat_mean = mean.flatten()
        block_cov = torch.block_diag(*cov)
        dist = MultivariateNormal(flat_mean, block_cov)
        samples = dist.sample((self.n_sample,))
        u_all = samples.view(self.n_sample, self.H, self.dof).clamp_(-self.cons_u, self.cons_u)
        return u_all
    
    def update_mean_cov(
        self,
        mu_prev: torch.Tensor,
        sigma_prev: torch.Tensor,
        x: torch.Tensor,
        u_all: torch.Tensor,
    ):
        w_cost = self.cost_fn(self.goal, x, u_all[:, 0, :])
        
        if self.store_samples:
            self.cost_all.append(w_cost.cpu().numpy())
            
        w = torch.exp(-self.beta * (w_cost - w_cost.min()))
        w_sum = w.sum()

        mu_new = (w[:, None, None] * u_all).sum(0) / w_sum
        diff = u_all - mu_new.unsqueeze(0)
        cov_w = torch.einsum('n,nhc,nhd->hcd', w, diff, diff) / w_sum

        mu_s = self.alpha_mean * mu_prev + (1 - self.alpha_mean) * mu_new
        Sigma_s = self.alpha_cov * sigma_prev + (1 - self.alpha_cov) * cov_w + self.cov_reg * torch.eye(self.dof, device=x.device)
        return mu_new[0], mu_s, Sigma_s
    
    
    def rollout(self, x_curr: Tensor):
        u_all = self.sample(self.mean, self.cov)
        best_u, self.mean, self.cov = self.update_mean_cov(self.mean, self.cov, x_curr, u_all)
        
        # Projection or Clipping input
        if self.proj_fn is not None:
            u_exec = self.proj_fn(best_u.unsqueeze(0), x_curr).squeeze(0)
        else:
            u_exec = best_u.clamp(-self.cons_u, self.cons_u)
        
        x_next = self.dynamics(x_curr, u_exec.unsqueeze(0), self.dt).squeeze(0)
        
        if self.store_samples:
            self.u_all_sample.append(u_all.cpu().numpy())
            self.q_all.append(x_next.cpu().numpy())
            self.u_exec_all.append(u_exec.cpu().numpy())
        
        return x_next
    
    def get_record(self):
        record = {
            "u_all_sample": np.array(self.u_all_sample) if self.store_samples else None,
            "cost_all": np.array(self.cost_all) if self.store_samples else None,
            "q_all": np.array(self.q_all) if self.store_samples else None,
            "u_exec_all": np.array(self.u_exec_all) if self.store_samples else None,
        }
        return record
      