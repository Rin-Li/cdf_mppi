import torch
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class CostConfig:
    q_min: torch.Tensor 
    q_max: torch.Tensor 
    sdf: Optional[Any] = None


class CostLinkRobotMPPI:
    def __init__(self, cfg: CostConfig, obs_list) -> None:
        self.q_min = cfg.q_min
        self.q_max = cfg.q_max
        self.sdf = cfg.sdf
        self.obs_list = obs_list
    

    def cost(self, x_ref: torch.Tensor, x: torch.Tensor, u_batch: torch.Tensor):
        """
        Evaluate the total cost for each trajectory.

        Args:
            all_traj: All sampled trajectories, shape (B, T, dof)
            closest_dist_all: Closest obstacle distances for each state in the trajectory, shape (B, T)

        Returns:
            total_cost: Vector of total costs for each trajectory, shape (B,)
        """
        closest_dist_all = self.sdf.inference_sdf(x, self.obs, return_grad=False)
        
        goal_cost = 10.0 * self.goal_cost(x[:, -1, :], x_ref)
        collision_cost = 100.0 * self.collision_cost(closest_dist_all)
        stagnation_cost = 10.0 * goal_cost * self.stagnation_cost(x)
        joint_limits_cost = 100.0 * self.joint_limits_cost(x)

        # Sum all cost components
        total_cost = goal_cost + collision_cost + joint_limits_cost + stagnation_cost
        return total_cost

    def goal_cost(self, traj_end, qf):
        """
        Compute Euclidean distance in joint space between the final state
        of the trajectory and the goal configuration.

        Args:
            traj_end: Final state of each trajectory, shape (B, dof)
            qf: Goal state, shape (dof,)

        Returns:
            Goal distance cost, shape (B,)
        """
        return (traj_end - qf).norm(p=2, dim=1)

    def collision_cost(self, closest_dist_all):
        """
        Penalize obstacle collisions using signed distance field (SDF) values.
        Only negative distances (penetration) contribute to the cost.

        Args:
            closest_dist_all: Closest distances to obstacles for each state, shape (B, T)

        Returns:
            Collision cost, shape (B,)
        """
        return torch.relu(-closest_dist_all).sum(dim=1)

    def joint_limits_cost(self, all_traj):
        """
        Penalize exceeding joint limits with a smooth squared penalty.

        Args:
            all_traj: Trajectories, shape (B, T, dof)

        Returns:
            Joint limit cost, shape (B,)
        """
        under = torch.relu(self.q_min - all_traj)
        over  = torch.relu(all_traj - self.q_max)
        return (under + over).pow(2).sum(dim=(1, 2))

    def stagnation_cost(self, all_traj):
        """
        Penalize trajectories that move very little from start to end.

        Args:
            all_traj: Trajectories, shape (B, T, dof)

        Returns:
            Stagnation cost, shape (B,)
        """
        dist = (all_traj[:, 0, :] - all_traj[:, -1, :]).norm(2, dim=1)
        return torch.clamp(1.0 / (dist + 1e-3), max=10.0)