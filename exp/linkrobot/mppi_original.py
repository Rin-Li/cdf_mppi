from planner.mppi import MPPI, MPPIConfig
import torch

class MPPILinkRobot:
    def __init__(self,
                start: torch.Tensor,
                goal: torch.Tensor, 
                mppi_cfg: MPPIConfig,
                cost_fn,
                dynamic_fn,
                proj_fn,
                sdf,
                obs_list):
    
        self.start = start.float()                        
        self.goal = goal.float()               
        self.obs_list = obs_list
        self.mppi = MPPI(mppi_cfg, dynamic_fn, cost_fn, proj_fn)
        self.sdf = sdf
        
        self.x_traj = [self.start.cpu().numpy().flatten()]
    
    def optimise(self, max_iter: int = 500):
        for k in range(max_iter):
            x_cur = torch.tensor(self.x_traj[-1], device=self.mppi.device, dtype=self.mppi.dtype)
            x_next = self.mppi.rollout(x_cur)
            # print(f"✅ Step {k+1}: from {x_cur.cpu().numpy()} to {x_next.cpu().numpy()}")
            h_val_next = self.sdf.inference_sdf(
                x_next, self.obs_list, return_grad=False
            ).item()
            
            if h_val_next < 0:
                print(f"❌ Step {k+1}: collision (h={h_val_next:.4f}) – abort.")
                return self.x_traj
            
            self.x_traj.append(x_next.cpu().numpy().flatten())
            if (x_next - self.goal).norm() < 0.1:
                break
            
        return self.x_traj

def main():
    from enviornment.linkrobot.robot2d_torch import Robot2D
    from enviornment.linkrobot.primitives2D_torch import Circle
    
    from exp.utils import InputBounds, dynamics, plot_2link_with_obstacle
    from planner.cost.mppi_cost_link_robot import CostLinkRobotMPPI, CostConfig
    from cdf.sdf2d import SDF2D
    import math
    
    PI = math.pi
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    robot = Robot2D()
    sdf = SDF2D(device, robot)
    circles = [
        Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3),
        Circle(center=torch.tensor([0.0, 2.45], device=device), radius=0.3),
    ]
    goal = torch.tensor([-2.1, 0.9], dtype=torch.float32, device=device)
    start = torch.tensor([[2.1, 1.2]], dtype=torch.float32, device=device)
    
    
    mppi_cfg = MPPIConfig(
        d_action=2,
        goal=goal,
        horizon=50,
        n_samples=200,
        dt=0.01,
        beta=1.0,
        u_clip=3.0,
        alpha_mean=0.5,
        alpha_cov=0.5,
        cov_reg=1e-3,
        device=device,
    )
    
    cost_cfg = CostConfig(
        q_min=torch.tensor([-PI, -PI], device=device),
        q_max=torch.tensor([PI, PI], device=device),
        sdf=sdf
    )
    
    cost_fn = CostLinkRobotMPPI(cost_cfg, circles).cost
    
    link_robot_planner = MPPILinkRobot(
        start=start.flatten(),
        goal=goal.flatten(),
        mppi_cfg=mppi_cfg,
        cost_fn=cost_fn,
        dynamic_fn=dynamics,
        proj_fn=None,
        sdf=sdf,
        obs_list=circles
    )
    
    obstacles = [(float(c.center[0]), float(c.center[1]), float(c.radius)) for c in circles]
    states = link_robot_planner.optimise(max_iter=5000)
    plot_2link_with_obstacle(states, obstacles=obstacles)

if __name__ == "__main__":
    main()
    
    