from planner.mppi import MPPI, MPPIConfig
from enviornment.franka.check_collision import initialize_pybullet, compute_min_distance_pybullet
import torch

class CDFMPPIFranka:
    def __init__(self,
                  start: torch.Tensor,
                  goal: torch.Tensor, 
                  mppi_cfg: MPPIConfig,
                  cost_fn,
                  dynamic_fn,
                  proj_fn,
                  cdf,
                  obs):
        
        self.start = start.float()                        
        self.goal = goal.float()
        self.start_np = self.start.cpu().numpy().flatten()               
        self.obs = obs
        self.mppi = MPPI(mppi_cfg, dynamic_fn, cost_fn, proj_fn)
        self.cdf = cdf
        self.robot_id, self.obs_ids, self.panda, self.link_indexes = initialize_pybullet(self.start_np, obs)
        
        self.x_traj = [self.start.cpu().numpy().flatten()]
    
    def optimise(self, max_iter: int = 500):
        for k in range(max_iter):
            x_cur = torch.tensor(self.x_traj[-1], device=self.mppi.device, dtype=self.mppi.dtype)
            x_next = self.mppi.rollout(x_cur)
            # print(f"✅ Step {k+1}: from {x_cur.cpu().numpy()} to {x_next.cpu().numpy()}")
            
            # Check collision using pybullet
            self.panda.hard_reset(x_next.detach().cpu().numpy())
            min_dist = compute_min_distance_pybullet(self.robot_id, self.obs_ids, self.link_indexes)
            
            if min_dist < 0:
                print(f"❌ step {k + 1}: collision (gap={min_dist}) – abort")
                return self.x_traj

            self.x_traj.append(x_next.cpu().numpy().flatten())
            if (x_next - self.goal).norm() < 0.03:
                break
            
        return self.x_traj

def main():
    from cdf.nn_cdf import CDF
    import numpy as np
    from enviornment.franka.obstacles import CrossObstacleManager
    from planner.cost.cdf_mppi_cost_franka import CostFrankaCDFMPPI, CostConfig
    from exp.utils import InputBounds, dynamics
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs = CrossObstacleManager(device=device)
    cdf = CDF(device=device)
    
    q0 = np.array([-1.57, 0.40000, 0.0000, -1.2708, 0.0000, 1.8675, 0.0000], dtype=np.float32)
    qf = np.array([ 1.57, 0.40000, 0.0000, -1.2708, 0.0000, 1.8675, 0.0000], dtype=np.float32)
    
    mppi_cfg = MPPIConfig(
        d_action=7,
        goal=qf,
        horizon=1,
        n_samples=200,
        dt=0.01,
        beta=1.0,
        u_clip=2.7,
        alpha_mean=0.5,
        alpha_cov=0.5,
        cov_reg=1e-3,
        device=device,
    )
    
    cost_cfg = CostConfig(
        alpha1=2000.0,
        alpha2=1000.0,
        activate=0.5,
        cdf=cdf
    )
    
    proj_fn = InputBounds(limits=[
        [-2.8973,  2.8973],
        [-1.7628,  1.7628],
        [-2.8973,  2.8973],
        [-3.0718, -0.0698],
        [-2.8973,  2.8973],
        [-0.0175,  3.7525],
        [-2.8973,  2.8973],
    ], dt=mppi_cfg.dt).project_to_step_box
    
    obs_pts = obs.pts()
    model = "cdf/model_dict.pt"
    
    cost_fn = CostFrankaCDFMPPI(cost_cfg, qf, obs_pts, model)
    
    
    franka_robot_planner = CDFMPPIFranka(
        start=torch.tensor(q0, device=device),
        goal=torch.tensor(qf, device=device),
        mppi_cfg=mppi_cfg,
        cost_fn=cost_fn.cost,
        dynamic_fn=dynamics,
        proj_fn=proj_fn,
        cdf=cdf,
        obs=obs
    )
    
    traj = franka_robot_planner.optimise(max_iter=500)
    
if __name__ == "__main__":
    main()
    
    
    
    