# Reference from https://github.com/yimingli1998/cdf

import torch
import math
from torchmin import minimize
import math
import copy

PI = math.pi

class CDF2D:
    def __init__(self,device, robot, q_max, q_min) -> None:
        '''
        Args:
            device: torch device
            robot: Robot2D instance
            q_max: maximum joint angles, tensor of shape (2,)
            q_min: minimum joint angles, tensor of shape (2,)
        '''
        self.device = device    
        self.link_length = torch.tensor([[2,2]]).float().to(device)
        self.num_joints = self.link_length.size(1)
        self.q_max = q_max
        self.q_min = q_min
        # robot
        self.robot = robot
        self.batchsize = 40000

    def inference_sdf(self,q,obj_lists,return_grad = False):  
        # using predefined object 
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
    
    def find_q(self,obj_lists,batchsize = None):
        # find q that makes d(x,q) = 0. x is the obstacle surface
        # using L-BFGS method
        if not batchsize:
            batchsize = self.batchsize
            
        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,2

            d = self.inference_sdf(q,obj_lists)
            cost = torch.sum(d**2)
            return cost
        
        # optimizer for data generation
        q = torch.rand(batchsize,2).to(self.device)*(self.q_max-self.q_min)+self.q_min
        q0 =copy.deepcopy(q)
        res = minimize(
            cost_function, 
            q, 
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
            )
        
        d = self.inference_sdf(q,obj_lists).squeeze()

        mask = torch.abs(d) < 0.05
        q_valid,d = res.x[mask],d[mask]
        boundary_mask = ((q_valid > self.q_min) & (q_valid < self.q_max)).all(dim=1)
        final_q = q_valid[boundary_mask]
        q0 = q0[mask][boundary_mask]
        # print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q),time.time()-t0))
        return q0,final_q,res.x

    def calculate_cdf(self,q,obj_lists,return_grad = False):
        # x : (Nx,2)
        # q : (Np,2)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0
        if not hasattr(self,'q_0_level_set'):
            self.q_0_level_set = self.find_q(obj_lists)[1]
        dist = torch.norm(q.unsqueeze(1) - self.q_0_level_set.unsqueeze(0),dim=-1)
  
        d = torch.min(dist,dim=-1)[0]
        # compute sign of d, based on the sdf
        # exit()
        d_ts = self.inference_sdf(q,obj_lists)

        mask =  (d_ts < 0)
        d[mask] = -d[mask]
        if return_grad:
            grad = torch.autograd.grad(d,q,torch.ones_like(d))[0]
            return d,grad
        return d 
    