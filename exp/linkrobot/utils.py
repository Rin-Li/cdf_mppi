import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle as mplCircle

def dynamics(x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    return x + u*dt      


def plot_2link_with_obstacle(states,
                              link_lengths=(2, 2),
                              obstacles=None,  # [(center_x, center_y, radius), ...]
                              show_traj=True):
    """
    states:      [N, 2] array of joint angles
    link_lengths: tuple (l1, l2)
    obstacles:   list of (center_x, center_y, radius)
    show_traj:   whether to plot the red trajectory line
    """
    l1, l2 = link_lengths
    traj = np.array(states)  # [N,2]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    if obstacles:
        for cx, cy, r in obstacles:
            obs_circle = mplCircle((cx, cy), r, color='gray', alpha=0.4)
            ax.add_patch(obs_circle)

    for q in traj:
        theta1, theta2 = q
        j1 = np.array([l1 * math.cos(theta1), l1 * math.sin(theta1)])
        j2 = j1 + np.array([l2 * math.cos(theta1 + theta2),
                            l2 * math.sin(theta1 + theta2)])
        plt.plot([0, j1[0], j2[0]], [0, j1[1], j2[1]], 'o-', alpha=0.3)

    if show_traj:
        plt.plot(traj[:, 0], traj[:, 1], 'r.-')

    plt.axis('equal')
    plt.grid(True)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    plt.show()


class InputBounds:
    def __init__(self, limits, dt: float):
        self.limits = limits
        self.dt = dt
        
    def step_u_bounds(self, q: torch.Tensor, limits, dt: float):
        """_summary_

        Args:
            q (torch.Tensor): _description_
            limits (_type_): _description_
            dt (float): _description_

        Returns:
            _type_: _description_
        """
        qmin = torch.tensor([limits[0][0], limits[1][0]], device=q.device, dtype=q.dtype)
        qmax = torch.tensor([limits[0][1], limits[1][1]], device=q.device, dtype=q.dtype)
        umin = (qmin - q) / dt
        umax = (qmax - q) / dt
        return umin, umax  # [2], [2]

    def project_to_step_box(self, u: torch.Tensor, q: torch.Tensor):
        """
        Project the control input u onto the feasible step box constraint:
        q + dt*u ∈ [qmin, qmax]
        Supported shapes:
        - u: [2]                 -> returns [2]
        - u: [N, H, 2] (H=1)     -> returns [N, H, 2]
        """
        umin, umax = self.step_u_bounds(q, self.limits, self.dt)  # [2]
        if u.dim() == 1:
            return torch.max(torch.min(u, umax), umin)

        return torch.max(torch.min(u, umax.view(1, 1, -1)),
                        umin.view(1, 1, -1))
        
