import torch
from enviornment.linkrobot.robot2d_torch import Robot2D
from enviornment.linkrobot.primitives2D_torch import Circle
from typing import List
import matplotlib.pyplot as plt

class SDF2D:
    def __init__(self, robot: Robot2D, device='cpu'):
        self.robot = robot
        self.device = device
        
    def inference_sdf(self, q, obj_lists : List[Circle],return_grad = False):  
        '''
        Args:
            q: Robot configuration (B, num_joints)
            obj_lists: List of Circle objects representing obstacles
            return_grad: If True, returns the gradient of SDF with respect to q
        Returns:
            sdf: Signed distance function values (B, N) where N is the number of obstacles
            grad: Gradient of SDF with respect to q (if return_grad is True)
        '''
        # using predefined object 
        if return_grad:
            q = q.clone().detach().requires_grad_(True)
        kpts = self.robot.surface_points_sampler(q)
        B,N = kpts.size(0),kpts.size(1)
        dist = torch.cat([obj.signed_distance(kpts.reshape(-1,2)).reshape(B,N,-1) for obj in obj_lists],dim=-1)

        # using closest point from robot surface
        sdf = torch.min(dist,dim=-1)[0]
        sdf = sdf.min(dim=-1)[0]
        if return_grad: 
            grad = torch.autograd.grad(sdf,q,torch.ones_like(sdf))[0]
            return sdf,grad
        return sdf

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    robot = Robot2D(num_joints=2, init_states=torch.tensor([[-1.6, -0.75]]), link_length=torch.tensor([[2, 2]]).float(), device=device)
    
    Circle1 = Circle(center=torch.tensor([2.0, 0.0]), radius=1.0, device=device)
    Circle2 = Circle(center=torch.tensor([0.0, 2.0]), radius=1.0, device=device)
    obj_lists = [Circle1, Circle2]
    sdf = SDF2D(robot, device='cpu')
    q = torch.tensor([[0, -0.75]])
    cur_f_rob = robot.forward_kinematics_all_joints(q)
    sdf_value, grad = sdf.inference_sdf(q, obj_lists, return_grad=True)
    # Plot robot
    print("SDF Value:", sdf_value)
    print("Gradient:", grad)
    _, ax = plt.subplots()
    ax.plot(cur_f_rob[0, 0, :], cur_f_rob[0, 1, :], color='blue', linewidth=1) 
    ax.axis("equal")
    for circle in obj_lists:
        circle.create_patch()
        ax.add_patch(circle.create_patch(color='red'))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.show()

if __name__ == "__main__":
    main()
    