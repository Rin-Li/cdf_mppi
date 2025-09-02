from planner.qp import QP, QPConfig
import torch

class QP2D:
    def __init__(self, qp_cfg: QPConfig, device: str, start: torch.Tensor, goal: torch.Tensor, cdf, obs_list):
        self.qp = QP(qp_cfg)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.cdf = cdf
        self.obs_list = obs_list
        self.goal = goal.float().to(self.device)
        
        self.start = start.float().to(self.device)
        
        self.x_traj = [self.start.clone()]
        
    def optimise(self, max_iter: int = 500):
        for k in range(max_iter):
            x_cur = self.x_traj[-1].clone().detach().requires_grad_(True)
            x_cur_for_cdf = x_cur.unsqueeze(0)
            
            
            distance, gradient = self.cdf.calculate_cdf(
                x_cur_for_cdf, 
                self.obs_list, 
                return_grad=True
            )
        
            x_cur_np = x_cur.detach().cpu().numpy().flatten()
            
            x_next_np = self.qp.optimise(x_cur_np, distance, gradient)
            
            x_next = torch.tensor(x_next_np.flatten(), device=self.device, dtype=self.dtype)
            
            x_next_for_cdf = x_next.unsqueeze(0)
                
            h_val_next = self.cdf.calculate_cdf(
                x_next_for_cdf, self.obs_list, return_grad=False
            )
            
            if isinstance(h_val_next, torch.Tensor):
                h_val_next = h_val_next.item() if h_val_next.numel() == 1 else h_val_next[0].item()
            
            if h_val_next < 0:
                print(f"❌ Step {k+1}: collision (h={h_val_next:.4f}) – abort.")
                return [x.cpu().numpy().flatten() for x in self.x_traj]
            
            self.x_traj.append(x_next.clone())
            
            if (x_next - self.goal).norm() < 0.1:
                break
            
        return [x.cpu().numpy().flatten() for x in self.x_traj]

def main():
    from enviornment.linkrobot.robot2d_torch import Robot2D
    from enviornment.linkrobot.primitives2D_torch import Circle
    
    from exp.utils import plot_2link_with_obstacle
    from cdf.cdf2d import CDF2D
    import math
    
    PI = math.pi
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    robot = Robot2D()
    q_max = torch.tensor([PI, PI]).to(device)
    q_min = torch.tensor([-PI, -PI]).to(device)
    cdf = CDF2D(device, robot, q_max, q_min)
    circles = [
        Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3),
        Circle(center=torch.tensor([0.0, 2.45], device=device), radius=0.3),
    ]
    goal = torch.tensor([-2.1, 0.9], dtype=torch.float32, device=device)
    start = torch.tensor([2.1, 1.2], dtype=torch.float32, device=device)
    
    qp_cfg = QPConfig(
        d_action=2,
        goal=goal.cpu().numpy(),
        dt=0.01,
        cons_u=3.0,
        planning_safety_buffer=0.6,
        Q_diag=[150.0, 190.0], 
        R_diag=[0.01, 0.01],
        x_box_limit=10.0,
    )
    
    link_robot_planner = QP2D(
        qp_cfg=qp_cfg,
        device=device,
        start=start,
        goal=goal,
        cdf=cdf,
        obs_list=circles
    )
    
    obstacles = [(float(c.center[0]), float(c.center[1]), float(c.radius)) for c in circles]
    x_traj = link_robot_planner.optimise(max_iter=500)
    plot_2link_with_obstacle(x_traj, obstacles=obstacles)
    
if __name__ == "__main__":
    main()