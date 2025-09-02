import torch
import math
from dataclasses import dataclass
from typing import Optional, Any

EPS = 1e-6
PI = math.pi

@dataclass
class CostConfig:
    alpha1: float = 2000.0
    alpha2: float = 1000.0
    activate: float = 1.0
    cdf: Optional[Any] = None
    

class CostFrankaCDFMPPI:
    def __init__(self, cfg: CostConfig, obs_pts, model) -> None:
        self.alpha1 = cfg.alpha1
        self.alpha2 = cfg.alpha2
        self.cdf = cfg.cdf
        self.activate = cfg.activate
        self.obs_pts = obs_pts
        self.net = model
        
    
    def cost(self,
                x_ref: torch.Tensor,   
                x:      torch.Tensor,   
                u_batch: torch.Tensor
                ):  
        
        x_req = x.detach().requires_grad_(True)
        
        h_val, h_dot = self.cdf.inference_d_wrt_q(self.obs_pts, x_req.unsqueeze(0), self.net, True)
        h_val  = h_val.item()
        n = h_dot.squeeze(0)
        if n.norm() > 1e-6:
            n = n / n.norm()

        move = u_batch                                
        ref = (x_ref - x).unsqueeze(0)               

        cos1 = (move * ref).sum(-1) / (move.norm(dim=-1)+EPS) / (ref.norm()+EPS)
        theta1 = torch.acos(cos1.clamp(-1.0, 1.0)) * 180.0 / PI
        cost = self.alpha2 * theta1          

        if h_val < self.activate and n.norm() > 0 and h_val < ref.norm():
            cos2 = (move * n).sum(-1) / (move.norm(dim=-1)+EPS) / (n.norm()+EPS)
            theta2 = torch.acos(cos2.clamp(-1.0, 1.0)) * 180.0 / PI
            penalty = torch.clamp(theta2 - 90.0, min=0.0)
            cost = cost + self.alpha1 * penalty             

        return cost                                  

    
    