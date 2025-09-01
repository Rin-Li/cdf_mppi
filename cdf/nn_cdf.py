# Reference from https://github.com/yimingli1998/cdf

import torch
from cdf.mlp import MLPRegression

class CDF:
    def __init__(self, device, model_path=None, model_index=None):
        self.device = device

        self.model = MLPRegression(
            input_dims=10, output_dims=1,
            mlp_layers=[1024, 512, 256, 128, 128],
            skips=[], act_fn=torch.nn.ReLU, nerf=True
        ).to(device)

        if model_path is not None:
            state = torch.load(model_path, map_location=device)
            if isinstance(state, dict) and all(isinstance(k, int) for k in state.keys()):
                self.model.load_state_dict(state[model_index])
            else:
                self.model.load_state_dict(state)
        self.model.eval()

    def inference(self, x, q):
        """Forward: (x:[Bx3], q:[Qx7]) → [B*Q,1]"""
        x, q = x.to(self.device), q.to(self.device)
        x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 3)
        q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, 7)
        inputs = torch.cat([x_cat, q_cat], dim=-1)
        return self.model.forward(inputs).reshape(len(x), len(q))

    def inference_d_wrt_q(self, x, q, return_grad=True):
        """_summary_

        Args:
            x : workspace points, (Bx3)
            q : robot config, (Qx7)
            return_grad (bool, optional): whether return gradient. Defaults to True.

        Returns:
            d : (Q) distance between q and x in C space.
            grad : (Qx7) gradient of d w.r.t q
        """
        q = q.clone().detach().requires_grad_(True).to(self.device)
        d_pred = self.inference(x, q)         
        d = d_pred.min(dim=0)[0]                
        if return_grad:
            grad = torch.autograd.grad(
                d, q, torch.ones_like(d),
                retain_graph=True, create_graph=True
            )[0]
            return d, grad
        else:
            return d
