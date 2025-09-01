import numpy as np
from dataclasses import dataclass
from typing import Tuple
import casadi as ca


@dataclass
class QPConfig:
    d_action: int
    goal: np.ndarray
    dt: float = 0.01
    cons_u: float = 3.0
    planning_safety_buffer: float = 0.6
    Q_diag: list[float] = [150, 190, 80, 70, 70, 90, 100]
    R_diag: list[float] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    x_box_limit: float = 10.0
    

class QP:
    def __init__(self, cfg: QPConfig):
        self.cfg = cfg
        self.dof = cfg.d_action
        self.dt = cfg.dt
        self.cons_u = cfg.cons_u
        self.planning_safety_buffer = cfg.planning_safety_buffer
        self.x_box_limit = cfg.x_box_limit

        self.Q = np.diag(cfg.Q_diag).astype(float)
        self.R= np.diag(cfg.R_diag).astype(float)
        self.A, self.B = self.create_system_matrices(self.dof, self.dt)

        self.goal = cfg.goal
        
    @staticmethod
    def create_system_matrices(n: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        A = np.eye(n, dtype=np.float32)
        B = np.eye(n, dtype=np.float32) * float(dt)
        return A, B
    
    def solve_qp(self, x0: np.ndarray, distance: float, gradient: np.ndarray):
        X = ca.MX.sym("X", self.dof, 2)   # k, k+1
        U = ca.MX.sym("U", self.dof, 1)

        goal = ca.DM(self.goal_np)
        cost = ca.mtimes((X[:, 1] - goal).T, ca.mtimes(self.Q, X[:, 1] - goal)) + ca.mtimes(U.T, ca.mtimes(R, U))

        # Constraints
        g = []
        g.append(X[:, 0] - x0.reshape(-1,))  # Xk equals x0
        g.append(X[:, 1] - (ca.mtimes(self.A, X[:, 0]) + ca.mtimes(self.B, U)))

        # Safety constraint
        grad_vec = np.array(gradient, dtype=float).reshape(-1)
        assert grad_vec.size == self.dof, f"gradient size {grad_vec.size} != {self.dof}"
        grad_row = ca.DM(grad_vec.reshape(1, self.dof))
        dist_val = float(distance)
        g.append(-ca.mtimes(grad_row, U) * self.dt - np.log(dist_val + float(self.planning_safety_buffer)))

        lbg = [0]*self.dof + [0]*self.dof + [-np.inf]
        ubg = [0]*self.dof + [0]*self.dof + [0]

        # Box bounds for X and U
        lbx = ca.vertcat(
            ca.repmat(-self.x_box_limit, self.dof),   # X[:,0]
            ca.repmat(-self.x_box_limit, self.dof),   # X[:,1]
            ca.repmat(-self.cons_u, self.dof),   # U
        )
        ubx = ca.vertcat(
            ca.repmat( self.x_box_limit, self.dof),
            ca.repmat( self.x_box_limit, self.dof),
            ca.repmat( self.cons_u, self.dof),
        )

        nlp = {
            "x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            "f": cost,
            "g": ca.vertcat(*g),
        }


        opts = {
            "print_time": False,
            "error_on_fail": False,
            "ipopt": {"print_level": 0, "sb": "yes"},
        }

        S = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = S(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

        xvec = sol["x"][:self.dof*2].full().reshape(2, self.dof).T  
        uvec = sol["x"][self.dof*2:].full().reshape(self.dof, 1)     
        return xvec.astype(np.float32), uvec.astype(np.float32)
    
    def optimise(self, q_curr)
        
    
    
    