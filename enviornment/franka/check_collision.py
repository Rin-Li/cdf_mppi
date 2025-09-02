# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Utility: Collision checking in PyBullet (headless)
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Tuple, List
import pybullet as p
import numpy as np
from pybullet_panda_sim import PandaSim, SphereManager


def _get_robot_id_fallback() -> int:
    """
    Choose the body with the most joints as 'the robot'.
    Raises RuntimeError if no bodies exist.
    """
    nb = p.getNumBodies()
    if nb == 0:
        raise RuntimeError("No bodies in the scene (cannot find robot_id).")
    return max(range(nb), key=lambda i: p.getNumJoints(i))


def initialize_pybullet(q0: np.ndarray, obs) -> Tuple[int, List[int]]:
    """
    Initialize PyBullet in DIRECT (headless) mode, spawn Panda and obstacles,
    and set robot to the initial joint configuration.

    Args:
        q0 (np.ndarray): initial 7-DoF joint angles (shape: (7,))
        obs (Obstacles): obstacle manager providing .as_spheres_numpy()

    Returns:
        robot_id (int): PyBullet body id of the robot
        obstacle_ids (list[int]): list of obstacle sphere ids
    """
    # Connect headless; if already connected, reset for a clean world.
    if p.isConnected():
        try:
            p.resetSimulation()
        except Exception:
            pass
    else:
        p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)

    # Spawn robot via PandaSim
    base_pos = [0, 0, 0]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    panda = PandaSim(p, base_pos, base_rot)

    # Try multiple attribute names, then fallback by scanning bodies.
    robot_id = getattr(panda, "robot_id", None) or getattr(panda, "robot", None)
    if robot_id is None or not isinstance(robot_id, int):
        robot_id = _get_robot_id_fallback()

    # Hard-set initial joints (instant set; unaffected by collisions/torque limits)
    panda.hard_reset(q0)

    # Obstacles → spheres
    obstacle_np = obs.as_spheres_numpy()   # (M, 4): x, y, z, r
    sphere_manager = SphereManager(p)
    # No rendering needed
    sphere_manager.initialize_spheres(obstacle_np, collide_only=False, visible=False)
    obstacle_ids = sphere_manager.body_ids
    num_joints = p.getNumJoints(robot_id)
    link_indices = [-1] + list(range(num_joints)) 

    return robot_id, obstacle_ids, panda, link_indices

def compute_min_distance_pybullet(
    robot_id: int,
    obstacle_ids: List[int],
    link_indices: List[int],
    dist_thresh: float = 5.0,
) -> float:
    """
    Query PyBullet closest points and return min contactDistance over all
    robot links vs. all obstacle bodies. If nothing within dist_thresh,
    returns dist_thresh.

    Args:
        robot_id (int): PyBullet body id of the robot
        obstacle_ids (list[int]): obstacle sphere ids
        link_indices (list[int]): robot link indices ([-1] + list(range(num_joints)))
        dist_thresh (float): query distance threshold

    Returns:
        float: minimum distance (negative if penetration)
    """
    min_d = float("inf")
    for obs_id in obstacle_ids:
        for link_idx in link_indices:
            cps = p.getClosestPoints(
                bodyA=robot_id,
                bodyB=obs_id,
                distance=dist_thresh,
                linkIndexA=link_idx,
                linkIndexB=-1,
            )
            if not cps:
                continue
            d_here = min(cp[8] for cp in cps)  # contactDistance
            if d_here < min_d:
                min_d = d_here
    if min_d == float("inf"):
        min_d = dist_thresh
    return float(min_d)
