import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from planner.mppi import MPPI, MPPIConfig
from enviornment.linkrobot.robot2d_torch import Robot2D
from enviornment.linkrobot.primitives2D_torch import Circle
from exp.utils import InputBounds, dynamics
from planner.cost.cdf_mppi_cost_link_robot import CostLinkRobotCDFMPPI, CostConfig
from cdf.cdf2d import CDF2D

PI = math.pi
L1, L2 = 2.0, 2.0
IMG_DIR = Path('img')

# paper-style colormap: blue (boundary/collision) → warm salmon (free space)
CMAP = LinearSegmentedColormap.from_list(
    'cdf',
    ['#4A87B9', '#8EB8D7', '#D4E5EE', '#F2D5B8', '#E8956D', '#C8442F'],
    N=256,
)
COLOR_OURS = '#C0392B'


def fk(q):
    # q: (2,) → (j0, j1, j2) world positions
    t1, t2 = q
    j0 = np.zeros(2)
    j1 = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
    j2 = j1 + np.array([L2 * np.cos(t1 + t2), L2 * np.sin(t1 + t2)])
    return j0, j1, j2


def compute_cdf_grid(cdf, circles, device, resolution=80):
    q1 = torch.linspace(-PI, PI, resolution, device=device)
    q2 = torch.linspace(-PI, PI, resolution, device=device)
    Q1, Q2 = torch.meshgrid(q1, q2, indexing='ij')
    q_flat = torch.stack([Q1.flatten(), Q2.flatten()], dim=1)
    chunks = []
    for i in range(0, len(q_flat), 2000):
        with torch.no_grad():
            chunks.append(cdf.calculate_cdf(q_flat[i:i+2000], circles).cpu())
    H = torch.cat(chunks).reshape(resolution, resolution).numpy()
    return Q1.cpu().numpy(), Q2.cpu().numpy(), H


def run_planner(device):
    robot  = Robot2D(device=device)
    q_max  = torch.tensor([PI,  PI], device=device)
    q_min  = torch.tensor([-PI, -PI], device=device)
    cdf    = CDF2D(device, robot, q_max, q_min)
    circles = [
        Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3, device=device),
        Circle(center=torch.tensor([0.0,  2.45], device=device), radius=0.3, device=device),
    ]
    goal  = torch.tensor([-2.1,  0.9], dtype=torch.float32, device=device)
    start = torch.tensor([ 2.1,  1.2], dtype=torch.float32, device=device)

    print('Computing q_0_level_set…')
    cdf.q_0_level_set = cdf.find_q(circles)[1]
    print(f'  {cdf.q_0_level_set.shape[0]} boundary points')

    proj_fn  = InputBounds(limits=[[-PI, PI], [-PI, PI]], dt=0.01).project_to_step_box
    cost_cfg = CostConfig(alpha1=20.0, alpha2=10.0, activate=0.5, cdf=cdf)
    cost_fn  = CostLinkRobotCDFMPPI(cost_cfg, circles).cost
    mppi_cfg = MPPIConfig(
        d_action=2, goal=goal, horizon=1, n_samples=200, dt=0.01,
        beta=1.0, u_clip=3.0, alpha_mean=0.5, alpha_cov=0.5, cov_reg=1e-3,
        device=device,
    )
    mppi = MPPI(mppi_cfg, dynamics, cost_fn, proj_fn)

    traj  = [start.cpu().numpy()]
    x_cur = start.clone()
    print('Running planner…')
    for k in range(5000):
        x_next = mppi.rollout(x_cur).reshape(-1)
        h = cdf.calculate_cdf(x_next.unsqueeze(0), circles).item()
        if h < 0:
            print(f'  Collision at step {k+1} (h={h:.4f})')
            break
        x_cur = x_next
        traj.append(x_cur.cpu().numpy())
        if (x_cur - goal).norm() < 0.1:
            print(f'  Goal reached at step {k+1}')
            break

    return np.array(traj), circles, cdf, start.cpu().numpy(), goal.cpu().numpy()


def save_static(traj, circles, cdf, start, goal, device):
    print('Computing CDF grid…')
    Q1, Q2, H = compute_cdf_grid(cdf, circles, device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # C-space
    ax = axes[0]
    H_disp = np.clip(H, 0, None)
    im = ax.pcolormesh(Q1, Q2, H_disp, cmap=CMAP, vmin=0, vmax=H_disp.max(),
                       shading='auto', rasterized=True)
    fig.colorbar(im, ax=ax)
    ax.contour(Q1, Q2, H, levels=[0], colors='k', linewidths=1.5)

    ax.plot(traj[:, 0], traj[:, 1], color=COLOR_OURS, linewidth=2,
            label='MPPI (Ours)', zorder=4)
    ax.plot(start[0], start[1], marker='o', markerfacecolor='white',
            markeredgecolor='k', markersize=9, zorder=5, label='Start')
    ax.plot(goal[0],  goal[1],  marker='*', markerfacecolor='#D4AC0D',
            markeredgecolor='k', markersize=15, zorder=5, label='Goal')

    ax.set_xlim(-PI, PI)
    ax.set_ylim(-PI, PI)
    ax.set_xlabel(r'$q_1$')
    ax.set_ylabel(r'$q_2$')
    ax.set_title('Configuration Space Trajectories')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax.set_aspect('equal')

    # Workspace
    ax = axes[1]
    for c in circles:
        cx, cy = c.center.cpu().numpy()
        ax.add_patch(plt.Circle((cx, cy), c.radius, color='#AAAAAA', alpha=0.6, zorder=2))

    eef = np.array([fk(q)[2] for q in traj])
    ax.plot(eef[:, 0], eef[:, 1], '--', color=COLOR_OURS, linewidth=2,
            label='MPPI (Ours)', zorder=4)

    j0s, j1s, j2s = fk(traj[0])
    ax.plot([j0s[0], j1s[0], j2s[0]], [j0s[1], j1s[1], j2s[1]],
            'o-', color='#555555', linewidth=2, markersize=5, zorder=3, label='Start pose')
    j0g, j1g, j2g = fk(traj[-1])
    ax.plot([j0g[0], j1g[0], j2g[0]], [j0g[1], j1g[1], j2g[1]],
            'o-', color=COLOR_OURS, linewidth=2, markersize=5, zorder=3, label='Goal pose')

    ax.plot(*fk(traj[0])[0], marker='o', markerfacecolor='white',
            markeredgecolor='k', markersize=9, zorder=6)
    ax.plot(*fk(traj[-1])[0], marker='*', markerfacecolor='#D4AC0D',
            markeredgecolor='k', markersize=15, zorder=6)

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Workspace Trajectories')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = IMG_DIR / 'demo_cspace_workspace.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()


def save_gif(traj, circles):
    print('Generating GIF…')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('CDF-MPPI — 2D Two-Link Robot')

    for c in circles:
        cx, cy = c.center.cpu().numpy()
        ax.add_patch(plt.Circle((cx, cy), c.radius, color='#AAAAAA', alpha=0.6))

    eef = np.array([fk(q)[2] for q in traj])
    trail, = ax.plot([], [], '--', color=COLOR_OURS, linewidth=1.5, alpha=0.8)
    arm,   = ax.plot([], [], 'o-', color='#2C7BB6', linewidth=2.5, markersize=6)

    indices = np.linspace(0, len(traj) - 1, min(120, len(traj)), dtype=int)

    def init():
        trail.set_data([], [])
        arm.set_data([], [])
        return trail, arm

    def update(fi):
        k = indices[fi]
        j0, j1, j2 = fk(traj[k])
        arm.set_data([j0[0], j1[0], j2[0]], [j0[1], j1[1], j2[1]])
        trail.set_data(eef[:k+1, 0], eef[:k+1, 1])
        return trail, arm

    ani = FuncAnimation(fig, update, frames=len(indices),
                        init_func=init, blit=True, interval=50)
    out = IMG_DIR / 'demo_animation.gif'
    ani.save(out, writer=PillowWriter(fps=20))
    print(f'Saved {out}')
    plt.close()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    traj, circles, cdf, start, goal = run_planner(device)
    print(f'Trajectory: {len(traj)} steps')
    save_static(traj, circles, cdf, start, goal, device)
    save_gif(traj, circles)
