# CDF-MPPI: Motion Planning with Configuration Space Distance Fields

## Advantages
- **Avoiding local optima**: The method eliminates optimization-based procedures, reducing the risk of convergence to local minima.  
- **Unified cost function**: An angle-based cost function integrates goal-seeking and obstacle avoidance into a single metric with consistent units.  
- **Computational efficiency**: By leveraging CDF gradients, the MPPI horizon can be reduced to one step ($H=1$), significantly lowering computation time while maintaining safety.  

---

## Method
We integrate **Configuration Space Distance Fields (CDFs)** with **Model Predictive Path Integral (MPPI)** control.  

### Robot dynamics
The joint configuration evolves as:
$$
\mathbf{q}_{t+1} = \mathbf{q}_t + \Delta t \cdot \mathbf{u},
$$
where $\mathbf{q}_t$ is the configuration and $\mathbf{u}$ the joint input.  

### Angle-based cost function
The cost is defined as:
$$
c(\theta_1, \theta_2) = \alpha_1 \theta_1 + \alpha_2 \theta_2,
$$
where:
- $\theta_1$: angle between the motion vector $\mathbf{q}_{t+1} - \mathbf{q}_t$ and the obstacle normal (from $\nabla_{\mathbf{q}} f_c$).  
- $\theta_2$: angle between the motion vector and the goal vector $\mathbf{q}_f - \mathbf{q}_t$.  

Explicitly:
$$
\theta_2(\mathbf{q}_t, \mathbf{q}_{t+1}) 
= \arccos \frac{(\mathbf{q}_{t+1} - \mathbf{q}_t) \cdot (\mathbf{q}_f - \mathbf{q}_t)}
{\lVert \mathbf{q}_{t+1} - \mathbf{q}_t \rVert \, \lVert \mathbf{q}_f - \mathbf{q}_t \rVert},
$$

$$
\theta_1^\star =
\arccos \frac{(\mathbf{q}_{t+1} - \mathbf{q}_t) \cdot \nabla_{\mathbf{q}} f_c(\mathcal{P}, \mathbf{q}_t)}
{\lVert \mathbf{q}_{t+1} - \mathbf{q}_t \rVert \, \lVert \nabla_{\mathbf{q}} f_c(\mathcal{P}, \mathbf{q}_t) \rVert}.
$$

$\theta_1$ is set to zero if the robot is far from obstacles or already moving away.  

### Joint limit projection
Sampled controls are projected into feasible ranges:
$$
\mathbf{u}_{\min}(\mathbf{q})=\frac{\mathbf{q}_{\min}-\mathbf{q}}{\Delta t}, \quad
\mathbf{u}_{\max}(\mathbf{q})=\frac{\mathbf{q}_{\max}-\mathbf{q}}{\Delta t}.
$$

The projection is:
$$
\mathrm{proj}_{\mathcal{U}_{\text{step}}(\mathbf{q})}(\mathbf{u})
=\operatorname{clip}\!\bigl(\mathbf{u};\,
\mathbf{u}_{\min}(\mathbf{q}),\,\mathbf{u}_{\max}(\mathbf{q})\bigr).
$$

---

## Results

### 2D Two-Link Robot
- Success rate: **99.2%**  
- Outperforms QP and original MPPI in robustness  
- Slightly longer but safer trajectories  

<p align="center">
  <img src="Exp1_trajectory.png" width="45%"/>
</p>

<p align="center">
  <em>Comparison of trajectories in configuration space (left) and workspace (right).</em>
</p>

| Method | Success Rate (%) | Path Length | Avg. Steps |
|--------|------------------|-------------|------------|
| MPPI (Ours) | 99.2 | 5.09 | 241 |
| MPPI (Original) | 71.2 | 4.40 | 155 |
| QP (IPOPT) | 64.1 | 3.75 | 107 |

---

### 7-DOF Franka Robot
- Achieves **>750 Hz** control frequency  
- Significantly higher success rate than QP and original MPPI in challenging environments  

<p align="center">
  <img src="Exp2_scene.png" width="45%"/>
</p>

<p align="center">
  <em>Comparison of trajectories in two scenarios with the 7-DOF Franka robot.</em>
</p>

| Method | Succ. (%) | Len. | Hz |
|--------|-----------|------|------|
| MPPI (Ours) | 100 / 83.8 | 4.57 / 7.85 | 776 / 718 |
| QP (IPOPT) | 13.0 / 14.0 | 4.30 / 6.74 | 222 / 228 |
| MPPI (Original) | - | - | 61 / 28 |

---
