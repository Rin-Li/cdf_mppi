"""
Benchmark CDF-MPPI on the 2D two-link robot.
Samples random start/goal pairs and reports success rate.
"""
import math
import torch
from planner.mppi import MPPI, MPPIConfig
from enviornment.linkrobot.robot2d_torch import Robot2D
from enviornment.linkrobot.primitives2D_torch import Circle
from exp.utils import InputBounds, dynamics
from planner.cost.cdf_mppi_cost_link_robot import CostLinkRobotCDFMPPI, CostConfig
from cdf.cdf2d import CDF2D

PI = math.pi
GOAL_TOL = 0.1
MAX_ITER = 5000
N_TRIALS = 250
MIN_DIST = 1.0   # min distance between start and goal in C-space


def make_mppi(goal, device):
    return MPPI(
        MPPIConfig(
            d_action=2,
            goal=goal,
            horizon=1,
            n_samples=200,
            dt=0.01,
            beta=1.0,
            u_clip=3.0,
            alpha_mean=0.5,
            alpha_cov=0.5,
            cov_reg=1e-3,
            device=device,
        ),
        dynamics,
        cost_fn=None,   # set after
        proj_fn=None,   # set after
    )


def run_trial(start, goal, cdf, circles, device):
    """Returns 'success', 'collision', or 'timeout'."""
    proj_fn = InputBounds(limits=[[-PI, PI], [-PI, PI]], dt=0.01).project_to_step_box
    cost_cfg = CostConfig(alpha1=20.0, alpha2=10.0, activate=0.5, cdf=cdf)
    cost_fn = CostLinkRobotCDFMPPI(cost_cfg, circles).cost

    cfg = MPPIConfig(
        d_action=2,
        goal=goal,
        horizon=1,
        n_samples=200,
        dt=0.01,
        beta=1.0,
        u_clip=3.0,
        alpha_mean=0.5,
        alpha_cov=0.5,
        cov_reg=1e-3,
        device=device,
    )
    mppi = MPPI(cfg, dynamics, cost_fn, proj_fn)

    x_cur = start.clone()
    for _ in range(MAX_ITER):
        x_next = mppi.rollout(x_cur).reshape(-1)   # always [dof]
        h = cdf.calculate_cdf(x_next.unsqueeze(0), circles).item()
        if h < 0:
            return 'collision'
        x_cur = x_next
        if (x_cur - goal).norm() < GOAL_TOL:
            return 'success'
    return 'timeout'


def sample_free(cdf, circles, device):
    """Sample a configuration that is collision-free (CDF > 0)."""
    while True:
        q = (torch.rand(1, 2, device=device) * 2 - 1) * PI
        h = cdf.calculate_cdf(q, circles).item()
        if h > 0.1:
            return q.squeeze(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    robot = Robot2D(device=device)
    q_max = torch.tensor([PI, PI], device=device)
    q_min = torch.tensor([-PI, -PI], device=device)
    cdf = CDF2D(device, robot, q_max, q_min)

    circles = [
        Circle(center=torch.tensor([2.3, -2.3], device=device), radius=0.3, device=device),
        Circle(center=torch.tensor([0.0, 2.45], device=device), radius=0.3, device=device),
    ]

    print("Pre-computing q_0_level_set (L-BFGS over 40k samples)...")
    cdf.find_q(circles)
    cdf.q_0_level_set = cdf.find_q(circles)[1]
    print(f"q_0_level_set size: {cdf.q_0_level_set.shape[0]} points")

    results = {'success': 0, 'collision': 0, 'timeout': 0}

    for i in range(N_TRIALS):
        start = sample_free(cdf, circles, device)
        goal = sample_free(cdf, circles, device)
        while (start - goal).norm() < MIN_DIST:
            goal = sample_free(cdf, circles, device)

        status = run_trial(start, goal, cdf, circles, device)
        results[status] += 1

        done = i + 1
        print(f"[{done:3d}/{N_TRIALS}] {status:10s} | "
              f"success={results['success']}/{done} "
              f"({100*results['success']/done:.1f}%)  "
              f"collision={results['collision']}  timeout={results['timeout']}", flush=True)

    print("\n=== Final Results ===")
    print(f"Success:   {results['success']}/{N_TRIALS} ({100*results['success']/N_TRIALS:.1f}%)")
    print(f"Collision: {results['collision']}/{N_TRIALS} ({100*results['collision']/N_TRIALS:.1f}%)")
    print(f"Timeout:   {results['timeout']}/{N_TRIALS} ({100*results['timeout']/N_TRIALS:.1f}%)")


if __name__ == "__main__":
    main()
