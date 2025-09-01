import torch
import numpy as np


class CrossObstacleManager:
    def __init__(self, sphere_radius: float = 0.05, samples_per_sphere: int = 40, device: str = "cuda"):
        self.sphere_radius = sphere_radius
        self.samples_per_sphere = samples_per_sphere
        self.device = torch.device(device)

        centers0 = self.pts()  
        self.centers0 = centers0.clone()  
        self.centers = centers0.clone()   

    def pts(
        self,
        center_x: float = 0.53,
        center_y: float = 0.0,
        center_z: float = 0.9,
        n_y: int = 15,
        n_z: int = 15,
        step_y: float = 0.02,
        step_z: float = None
    ) -> torch.Tensor:
        device = getattr(self, "device", "cpu")
        dtype = torch.float32
        if step_z is None:
            step_z = step_y
        n_y = max(1, int(n_y))
        n_z = max(1, int(n_z))

        cx = torch.tensor(center_x, device=device, dtype=dtype)
        cy = torch.tensor(center_y, device=device, dtype=dtype)
        cz = torch.tensor(center_z, device=device, dtype=dtype)

        offs_y = (torch.arange(n_y, device=device, dtype=dtype) - (n_y - 1) / 2.0) * float(step_y)
        offs_z = (torch.arange(n_z, device=device, dtype=dtype) - (n_z - 1) / 2.0) * float(step_z)

        centers_y = torch.stack([
            torch.full((n_y,), cx, device=device, dtype=dtype),
            cy + offs_y,
            torch.full((n_y,), cz, device=device, dtype=dtype)
        ], dim=-1)

        centers_z = torch.stack([
            torch.full((n_z,), cx, device=device, dtype=dtype),
            torch.full((n_z,), cy, device=device, dtype=dtype),
            cz + offs_z
        ], dim=-1)

        centers = torch.cat([centers_y, centers_z], dim=0)  # 保持 [水平, 垂直] 的固定顺序
        return centers

    # -------- 为 CDF 构造点云：不依赖 unique ----------
    def cross_spheres_points_for_cdf(self, sphere_centers: torch.Tensor) -> torch.Tensor:
        all_pts = []
        for c in sphere_centers:
            pts = self.fibonacci_sphere_points(self.samples_per_sphere, c)
            all_pts.append(pts)
        return torch.cat(all_pts, dim=0)

    # -------- 基于 step 的“论文式”运动：纯刚体平移 ----------
    def move_pts_step(
        self,
        step: int,
        dz: float = 0.01,
        thr_bottom: float = 0.45,
        reset_z: float = 0.65,
        shift_x: float = 0.02
    ) -> torch.Tensor:

        c0 = self.centers0  

        mean_z0 = float(c0[:, 2].mean().item())

        cycle_len = max(1, int(np.floor((mean_z0 - thr_bottom) / dz)))

        step_in_cycle = step % cycle_len
        n_cycles = step // cycle_len

        z_offset = - step_in_cycle * dz

        x_offset = n_cycles * shift_x

        offset = torch.tensor([x_offset, 0.0, z_offset], device=self.device, dtype=torch.float32)

        centers = c0 + offset

        if step_in_cycle == cycle_len - 1:
            mean_now = float(centers[:, 2].mean().item())
            centers[:, 2] += (reset_z - mean_now)

        self.centers = centers
        return centers

    def as_spheres_numpy(self, r: float = None) -> np.ndarray:
        if r is None:
            r = self.sphere_radius
        pts = self.centers.detach().cpu()
        radii = torch.full((pts.shape[0], 1), float(r))
        return torch.cat([pts, radii], dim=1).numpy()

