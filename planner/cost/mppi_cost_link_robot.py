import torch
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class CostConfig:
    q_min: torch.Tensor
    q_max: torch.Tensor
    dt: float = 0.01
    sdf: Optional[Any] = None


class CostLinkRobotMPPI:
    def __init__(self, cfg: CostConfig, obs_list) -> None:
        self.q_min = cfg.q_min
        self.q_max = cfg.q_max
        self.dt = cfg.dt
        self.sdf = cfg.sdf
        self.obs_list = obs_list

    def cost(self, x_ref: torch.Tensor, x: torch.Tensor, u_batch: torch.Tensor):
        """
        Args:
            x_ref:   goal configuration, shape (dof,)
            x:       current configuration, shape (dof,)
            u_batch: sampled controls, shape (N, dof)
        Returns:
            total cost per sample, shape (N,)
        """
        q_next = x.unsqueeze(0) + u_batch * self.dt  # (N, dof)

        goal_cost = torch.norm(q_next - x_ref.unsqueeze(0), dim=-1)  # (N,)

        sdf_vals = self.sdf.inference_sdf(q_next, self.obs_list)  # (N,)
        collision_cost = torch.relu(-sdf_vals + 0.1)

        under = torch.relu(self.q_min.unsqueeze(0) - q_next)
        over = torch.relu(q_next - self.q_max.unsqueeze(0))
        joint_cost = (under + over).pow(2).sum(dim=1)

        return goal_cost + 100.0 * collision_cost + 100.0 * joint_cost
