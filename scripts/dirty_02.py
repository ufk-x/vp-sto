"""
基于123.py的双平面约束脚本
在原有垂直平面约束的基础上，添加一个水平平面约束
- 平面1：垂直于xoy平面，通过起点和终点连线
- 平面2：平行于xoy平面，通过起点和终点连线的中点
这样的双约束应该能让末端执行器沿直线运动
"""

import sys
import time
import numpy as np
import pybullet as p
import pinocchio as pin
from config import PLANNER_PATH
from config import PANDA_URDF
sys.path.append(PLANNER_PATH)

from benchmarks.playground.env_simple_world import SimpleWorld
from benchmarks.utils import (
    get_joint_limits, get_max_velocity, get_dynamical_limits,
    # 下面这些是为可视化新增：
    draw_point, add_line, GREEN                      # 可视化用
)

from planners.vptraj import VPTraj
from matplotlib import pyplot as plt

# ========================
# PARAMETERS
# ========================
qg = np.array([0, 0, 0, -np.pi/2, 0, np.pi/1, np.pi/4], dtype=float)
dq0 = np.zeros(7, dtype=float)
dqg = np.zeros(7, dtype=float)

# 采样/代价参数
R_sampling = 1e1
Q_min = 1e0
Q_max = 1e3
factor_Q_min = 1e-1
factor_Q_max = 1e1

N_via = 4
N_candidates = 100
N_eval = 50
dt_control = 0.01        # 控制周期（s）
sim_duration = 5.0       # 仿真总时长（s）

true_amax_default = 3.0  # rad/s^2（示意）
gripper_opening = 0.06   # 夹爪开口

# ===== 可视化参数（你可以随时改） =====
DRAW_GOAL_AXES = True         # 是否画目标位姿坐标架
DRAW_GOAL_POINT = True        # 是否画目标位置点标
GOAL_POINT_SIZE = 0.02
DRAW_EEF_TRAIL = True         # 是否画末端轨迹
TRAIL_LIFETIME = 0.0          # 线段寿命（秒），0=永久
TRAIL_STRIDE = 1              # 每多少个控制周期画一次线（>=1）

# ===== 平面可视化参数 =====
DRAW_CONSTRAINT_PLANES = True  # 是否画约束平面
PLANE_SIZE = 0.5              # 平面半尺寸 (m)
PLANE_ALPHA = 0.3             # 平面透明度
VERTICAL_PLANE_COLOR = [0.8, 0.8, 0.2]  # 垂直平面颜色 (黄色)
HORIZONTAL_PLANE_COLOR = [0.2, 0.8, 0.8]  # 水平平面颜色 (青色)
PLANE_GRID_MODE = True        # 是否使用网格模式（更清晰）
PLANE_GRID_SIZE = 8           # 网格密度

LOSS_GOAL = 1e5
# ===== 双平面约束参数 =====
USE_DUAL_PLANE_CONSTRAINT = True    # 是否启用双平面约束
VERTICAL_CONSTRAINT_WEIGHT = 1e6    # 垂直平面约束惩罚权重
HORIZONTAL_CONSTRAINT_WEIGHT = 1e6  # 水平平面约束惩罚权重
PLANE_TOLERANCE = 0.01              # 平面约束容忍度 (m)


# ========================
# Pinocchio FK Helper
# ========================
class PinocchioFK:
    """Pinocchio前向运动学计算器"""
    def __init__(self, urdf_path, end_effector_name="panda_grasptarget"):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 查找末端执行器link ID
        try:
            self.ee_frame_id = self.model.getFrameId(end_effector_name)
        except:
            # 如果找不到frame，尝试link
            try:
                self.ee_frame_id = self.model.getJointId(end_effector_name)
            except:
                # 默认使用最后一个frame
                self.ee_frame_id = len(self.model.frames) - 1
                print(f"[WARNING] Could not find {end_effector_name}, using frame {self.ee_frame_id}")
        
        print(f"[INFO] Pinocchio model loaded with {self.model.nq} DOF")
        print(f"[INFO] End-effector frame ID: {self.ee_frame_id}")
    
    def compute_fk(self, q):
        """
        计算前向运动学
        Args:
            q: 关节角度 [7,] 或 [N, 7] (只包含手臂关节)
        Returns:
            positions: 末端执行器位置 [3,] 或 [N, 3]
        """
        q = np.asarray(q, dtype=float)
        
        # 扩展为完整的关节配置（添加夹爪关节）
        if q.ndim == 1:
            # 单个配置：7 -> 9（添加两个夹爪关节，默认值为0.04）
            if q.shape[0] == 7:
                q_full = np.concatenate([q, [0.04, 0.04]])
            elif q.shape[0] == 9:
                q_full = q
            else:
                raise ValueError(f"Expected 7 or 9 DOF, got {q.shape[0]}")
            
            pin.framesForwardKinematics(self.model, self.data, q_full)
            pos = self.data.oMf[self.ee_frame_id].translation.copy()
            return pos
        
        elif q.ndim == 2:
            # 批量配置
            if q.shape[1] == 7:
                # 7 DOF -> 9 DOF
                gripper_joints = np.full((q.shape[0], 2), 0.04)
                q_full = np.concatenate([q, gripper_joints], axis=1)
            elif q.shape[1] == 9:
                q_full = q
            else:
                raise ValueError(f"Expected 7 or 9 DOF, got {q.shape[1]}")
            
            positions = np.zeros((q.shape[0], 3))
            for i in range(q.shape[0]):
                pin.framesForwardKinematics(self.model, self.data, q_full[i])
                positions[i] = self.data.oMf[self.ee_frame_id].translation.copy()
            return positions
        
        else:
            raise ValueError("q must be 1D or 2D array")
    
    def compute_fk_batch_3d(self, q_batch):
        """
        计算3D批量前向运动学 [N_candidates, N_timesteps, 7] -> [N_candidates, N_timesteps, 3]
        """
        N_candidates, N_timesteps, dof = q_batch.shape
        positions = np.zeros((N_candidates, N_timesteps, 3))
        
        # 处理关节维度
        if dof == 7:
            # 7 DOF -> 9 DOF（添加夹爪关节）
            gripper_joints = np.full((N_candidates, N_timesteps, 2), 0.04)
            q_full = np.concatenate([q_batch, gripper_joints], axis=2)
        elif dof == 9:
            q_full = q_batch
        else:
            raise ValueError(f"Expected 7 or 9 DOF, got {dof}")
        
        for i in range(N_candidates):
            for j in range(N_timesteps):
                pin.framesForwardKinematics(self.model, self.data, q_full[i, j])
                positions[i, j] = self.data.oMf[self.ee_frame_id].translation.copy()
        
        return positions


# ========================
# Dual Plane Constraint Functions
# ========================
def compute_dual_plane_constraints(start_pos, goal_pos):
    """
    计算双平面约束参数
    - 垂直平面：垂直于xoy平面，通过起点和终点连线
    - 水平平面：平行于xoy平面，通过起点和终点连线
    
    Args:
        start_pos: 起始位置 [x, y, z]
        goal_pos: 目标位置 [x, y, z]
    
    Returns:
        constraint_params: 约束参数字典，包含两个平面
    """
    start_pos = np.array(start_pos, dtype=float)
    goal_pos = np.array(goal_pos, dtype=float)
    
    # 连线向量
    direction = goal_pos - start_pos
    length = np.linalg.norm(direction)
    
    if length < 1e-6:
        # 起点终点重合
        return {"type": "none"}
    
    direction_unit = direction / length
    
    # === 垂直平面约束 ===
    # 平面法向量：连线在xy平面的投影的垂直方向
    direction_xy = direction[:2]  # [dx, dy]
    length_xy = np.linalg.norm(direction_xy)
    
    if length_xy < 1e-6:
        # 连线完全垂直，垂直平面约束失效
        vertical_constraint = {"type": "none"}
    else:
        # 垂直平面法向量在xy平面内，垂直于连线投影
        normal_xy = np.array([-direction_xy[1], direction_xy[0]]) / length_xy
        vertical_normal = np.array([normal_xy[0], normal_xy[1], 0.0])  # z分量为0
        
        vertical_constraint = {
            "type": "plane",
            "normal": vertical_normal,
            "point": start_pos,
            "name": "vertical"
        }
    
    # === 水平平面约束 ===
    # 平面法向量：z方向单位向量 [0, 0, 1]
    horizontal_normal = np.array([0.0, 0.0, 1.0])
    
    # 平面通过起点和终点连线的中点（或者可以选择起点或终点）
    middle_point = (start_pos + goal_pos) / 2
    
    horizontal_constraint = {
        "type": "plane", 
        "normal": horizontal_normal,
        "point": middle_point,  # 也可以用start_pos或goal_pos
        "name": "horizontal"
    }
    
    return {
        "type": "dual_plane",
        "vertical": vertical_constraint,
        "horizontal": horizontal_constraint,
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "direction": direction_unit,
        "length": length
    }

def compute_dual_plane_constraint_violation(positions, constraint_params):
    """
    计算双平面约束违反程度
    
    Args:
        positions: 末端执行器位置数组，形状 [N_candidates, N_timesteps, 3]
        constraint_params: 双平面约束参数
    
    Returns:
        violations: 违反程度字典，包含垂直和水平平面的违反情况
            - "vertical": [N_candidates, N_timesteps]
            - "horizontal": [N_candidates, N_timesteps]
            - "total": [N_candidates, N_timesteps]
    """
    if constraint_params["type"] != "dual_plane":
        return {
            "vertical": np.zeros(positions.shape[:2]),
            "horizontal": np.zeros(positions.shape[:2]),
            "total": np.zeros(positions.shape[:2])
        }
    
    # 垂直平面约束违反
    vertical_violations = np.zeros(positions.shape[:2])
    if constraint_params["vertical"]["type"] == "plane":
        normal = constraint_params["vertical"]["normal"]
        point = constraint_params["vertical"]["point"]
        
        # 点到平面距离
        PP0 = positions - point[None, None, :]  # [N_candidates, N_timesteps, 3]
        vertical_violations = np.abs(np.sum(PP0 * normal[None, None, :], axis=2))
    
    # 水平平面约束违反
    horizontal_violations = np.zeros(positions.shape[:2])
    if constraint_params["horizontal"]["type"] == "plane":
        normal = constraint_params["horizontal"]["normal"]
        point = constraint_params["horizontal"]["point"]
        
        # 点到平面距离
        PP0 = positions - point[None, None, :]  # [N_candidates, N_timesteps, 3]
        horizontal_violations = np.abs(np.sum(PP0 * normal[None, None, :], axis=2))
    
    # 总违反程度（可以用不同的组合方式）
    total_violations = vertical_violations + horizontal_violations
    
    return {
        "vertical": vertical_violations,
        "horizontal": horizontal_violations,
        "total": total_violations
    }


# ========================
# Dual Plane Visualization Helpers
# ========================
def draw_dual_constraint_planes(constraint_params, size=0.5, 
                               vertical_color=[0.8, 0.8, 0.2], horizontal_color=[0.2, 0.8, 0.8],
                               grid_mode=True, grid_size=8):
    """
    在PyBullet中绘制双约束平面
    
    Args:
        constraint_params: 双平面约束参数
        size: 平面半尺寸
        vertical_color: 垂直平面颜色
        horizontal_color: 水平平面颜色
        grid_mode: 是否使用网格模式
        grid_size: 网格密度
    """
    if constraint_params["type"] != "dual_plane":
        return None
    
    start_pos = constraint_params["start_pos"]
    goal_pos = constraint_params["goal_pos"]
    
    # 绘制垂直平面
    if constraint_params["vertical"]["type"] == "plane":
        draw_single_constraint_plane(
            constraint_params["vertical"], start_pos, goal_pos,
            size=size, color=vertical_color, 
            grid_mode=grid_mode, grid_size=grid_size,
            plane_name="Vertical"
        )
    
    # 绘制水平平面
    if constraint_params["horizontal"]["type"] == "plane":
        draw_single_constraint_plane(
            constraint_params["horizontal"], start_pos, goal_pos,
            size=size, color=horizontal_color,
            grid_mode=grid_mode, grid_size=grid_size,
            plane_name="Horizontal"
        )
    
    print(f"[INFO] Dual constraint planes visualized")

def draw_single_constraint_plane(plane_params, start_pos, goal_pos, size=0.5, color=[0.8, 0.8, 0.2], 
                                grid_mode=True, grid_size=8, plane_name=""):
    """
    绘制单个约束平面
    """
    normal = plane_params["normal"]  # 平面法向量
    point = plane_params["point"]    # 平面上一点
    
    # 计算平面中心点（起点和终点的中点）
    center = (start_pos + goal_pos) / 2
    
    # 确保中心点在平面上（投影到平面）
    distance_to_plane = np.dot(normal, center - point)
    center_on_plane = center - distance_to_plane * normal
    
    # 构造平面的两个正交方向向量
    direction = goal_pos - start_pos
    direction_length = np.linalg.norm(direction)
    if direction_length < 1e-6:
        return None
    
    u_direction = direction / direction_length  # 连线方向单位向量
    
    # 选择与法向量正交的方向向量
    if plane_params.get("name") == "horizontal":
        # 水平平面：法向量是[0,0,1]，选择连线方向作为u，垂直方向作为v
        v_direction = np.cross(normal, u_direction)  # 应该在xy平面内
        if np.linalg.norm(v_direction) < 1e-6:
            # 如果连线方向与法向量平行，选择x方向
            v_direction = np.array([1, 0, 0])
            u_direction = np.cross(v_direction, normal)
        v_direction = v_direction / np.linalg.norm(v_direction)
    else:
        # 垂直平面：选择z方向和连线方向
        v_direction = np.array([0, 0, 1])  # z方向
        # 如果连线已经是z方向，选择另一个方向
        if abs(np.dot(u_direction, v_direction)) > 0.99:
            v_direction = np.array([1, 0, 0])  # x方向
    
    if grid_mode:
        # 网格模式：绘制详细网格
        step_u = 2 * size / grid_size
        step_v = 2 * size / grid_size
        line_color = [c * 0.8 for c in color]
        
        # U方向的网格线
        for i in range(grid_size + 1):
            v_offset = -size + i * step_v
            start_grid = center_on_plane - size * u_direction + v_offset * v_direction
            end_grid = center_on_plane + size * u_direction + v_offset * v_direction
            width = 2 if i == 0 or i == grid_size or i == grid_size//2 else 1
            add_line(start_grid, end_grid, color=line_color, width=width, lifetime=0.0)
        
        # V方向的网格线
        for i in range(grid_size + 1):
            u_offset = -size + i * step_u
            start_grid = center_on_plane + u_offset * u_direction - size * v_direction
            end_grid = center_on_plane + u_offset * u_direction + size * v_direction
            width = 2 if i == 0 or i == grid_size or i == grid_size//2 else 1
            add_line(start_grid, end_grid, color=line_color, width=width, lifetime=0.0)
    
    # 绘制平面边框（加粗）
    corners = [
        center_on_plane - size * u_direction - size * v_direction,
        center_on_plane + size * u_direction - size * v_direction,
        center_on_plane + size * u_direction + size * v_direction,
        center_on_plane - size * u_direction + size * v_direction
    ]
    
    border_color = [c * 0.5 for c in color]  # 边框颜色更深
    for i in range(4):
        next_i = (i + 1) % 4
        add_line(corners[i], corners[next_i], color=border_color, width=3, lifetime=0.0)
    
    # 绘制法向量（从中心点出发）
    normal_end = center_on_plane + 0.15 * normal
    add_line(center_on_plane, normal_end, color=[1, 0, 1], width=4, lifetime=0.0)  # 紫色法向量
    
    # 在法向量末端画个小球表示方向
    draw_point(normal_end, size=0.01, color=[1, 0, 1])
    
    print(f"[INFO] {plane_name} constraint plane visualized:")
    print(f"  - Center: {center_on_plane}")
    print(f"  - Normal: {normal}")
    print(f"  - Size: {size}m x {size}m")


# ========================
# Controller
# ========================
class DualPlaneConstraintController7D:
    def __init__(self, N_eval, N_via, vel_lim, acc_lim, q_limits, qg,
                 dt_control, N_candidates, R_sampling, pinocchio_fk=None, constraint_params=None):
        self.ndof = 7
        self.vptraj = VPTraj(ndof=self.ndof, N_eval=N_eval, N_via=N_via,
                             vel_lim=np.asarray(vel_lim, dtype=float),
                             acc_lim=np.asarray(acc_lim, dtype=float))
        self.vptraj_idle = VPTraj(ndof=self.ndof, N_eval=N_eval, N_via=1,
                                  vel_lim=np.asarray(vel_lim, dtype=float),
                                  acc_lim=np.asarray(acc_lim, dtype=float))

        self.q_limits = np.asarray(q_limits, dtype=float)
        self.qg = np.asarray(qg, dtype=float)
        self.dt_control = float(dt_control)
        self.N_candidates = int(N_candidates)
        self.R = float(R_sampling)
        
        # 双平面约束相关 - 使用Pinocchio FK
        self.pinocchio_fk = pinocchio_fk  # PinocchioFK实例
        self.constraint_params = constraint_params or {"type": "none"}

        self.Q = Q_max
        self.Q_log = []
        self.p_next = None
        self.T_next = None
        self.samples_log = []
        self.samples_loss_log = []
        self.sol_log = []

    def reset(self):
        self.Q = Q_max
        self.Q_log.clear()
        self.p_next = None
        self.T_next = None
        self.samples_log.clear()
        self.samples_loss_log.clear()
        self.sol_log.clear()

    def loss_fn(self, q, dq, ddq, T):
        T = np.asarray(T)
        duration_cost = np.full(q.shape[0], T) if T.ndim == 0 else T
        qT = q[:, -1, :]
        terminal_cost = LOSS_GOAL * np.sum((qT - self.qg)**2, axis=1)

        low = self.q_limits[:, 0][None, None, :]
        up  = self.q_limits[:, 1][None, None, :]
        viol = (q < low) | (q > up)
        limit_violation_cost = 1e6 * np.sum(viol, axis=(1, 2))

        # 双平面约束惩罚
        dual_plane_constraint_cost = 0.0
        if USE_DUAL_PLANE_CONSTRAINT and self.pinocchio_fk is not None and self.constraint_params["type"] == "dual_plane":
            # 计算轨迹上每个点的末端执行器位置
            eef_positions = self._compute_eef_positions_batch(q)  # [N_candidates, N_timesteps, 3]
            
            # 计算双平面约束违反
            violations = compute_dual_plane_constraint_violation(eef_positions, self.constraint_params)
            
            # 计算惩罚
            tolerance = PLANE_TOLERANCE
            
            # 垂直平面违反惩罚
            vertical_excess = np.maximum(0, violations["vertical"] - tolerance)
            vertical_cost = VERTICAL_CONSTRAINT_WEIGHT * np.sum(vertical_excess**2, axis=1)
            
            # 水平平面违反惩罚
            horizontal_excess = np.maximum(0, violations["horizontal"] - tolerance)
            horizontal_cost = HORIZONTAL_CONSTRAINT_WEIGHT * np.sum(horizontal_excess**2, axis=1)
            
            dual_plane_constraint_cost = vertical_cost + horizontal_cost

        return terminal_cost + duration_cost + limit_violation_cost + dual_plane_constraint_cost
    
    def _compute_eef_positions_batch(self, q_batch):
        """
        批量计算末端执行器位置 - 使用Pinocchio
        
        Args:
            q_batch: 关节角度，形状 [N_candidates, N_timesteps, 7]
        
        Returns:
            eef_positions: 末端位置，形状 [N_candidates, N_timesteps, 3]
        """
        if self.pinocchio_fk is None:
            # 如果没有Pinocchio FK，返回零矩阵
            return np.zeros((q_batch.shape[0], q_batch.shape[1], 3))
        
        return self.pinocchio_fk.compute_fk_batch_3d(q_batch)

    def predictive_sampling(self, q, dq):
        if len(self.samples_loss_log) > 0:
            num_viol = np.sum(self.samples_loss_log[-1] > 1e6)
        else:
            num_viol = 0
        ratio = num_viol / max(1, self.N_candidates)
        self.Q *= np.clip(np.exp(-3 * (ratio - 0.5)), factor_Q_min, factor_Q_max)
        self.Q = np.clip(self.Q, Q_min, Q_max)
        self.Q_log.append(self.Q)

        pos, vel, acc, p, T = self.vptraj.sample_trajectories(
            self.N_candidates, q, dq0=dq, qT=self.qg, dqT=np.zeros_like(dq),
            Q=self.Q, R=self.R
        )
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], T)
        self.samples_log.append(pos)
        self.samples_loss_log.append(loss)
        i_best = int(np.argmin(loss))
        return p[i_best], loss[i_best], float(T[i_best])

    def previous_sol(self, q, dq):
        if self.p_next is None:
            return None, np.inf, 0.0
        pos, vel, acc = self.vptraj.get_trajectory(self.p_next, q, dq0=dq,
                                                   dqT=np.zeros_like(dq), T=self.T_next)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [self.T_next])[0]
        return self.p_next, loss, self.T_next

    def idle(self, q, dq):
        acc_lim = self.vptraj_idle.acc_lim
        T_idle = float(np.max(np.divide(np.abs(dq), np.maximum(acc_lim, 1e-6))))
        q_idle = q + 0.5 * dq * T_idle
        pos, vel, acc = self.vptraj_idle.get_trajectory(q_idle, q, dq0=dq,
                                                        dqT=np.zeros_like(dq), T=T_idle)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [T_idle])[0]
        if T_idle < self.dt_control:
            return q_idle, np.zeros_like(dq), loss, 0.0
        q_next, dq_next, _ = self.vptraj_idle.get_trajectory_at_time(
            self.dt_control, q_idle, q, dq0=dq, dqT=np.zeros_like(dq), T=T_idle
        )
        return q_next.squeeze(), dq_next.squeeze(), loss, T_idle

    def control(self, q, dq):
        q_idle, dq_idle, loss_idle, T_idle = self.idle(q, dq)
        p_prev, loss_prev, T_prev = self.previous_sol(q, dq)
        p_samp, loss_samp, T_samp = self.predictive_sampling(q, dq)

        choose_idle = (loss_idle <= loss_prev) and (loss_idle <= loss_samp)
        choose_prev = (loss_prev <= loss_samp)

        if choose_idle:
            print(f"[Idle ] loss={loss_idle:.2f}", end="\r")
            self.p_next, self.T_next = None, None
            self.sol_log.append(np.vstack((q, q_idle)))
            return q_idle, dq_idle

        if choose_prev:
            print(f"[Prev ] loss={loss_prev:.2f}", end="\r")
            p_best, T_best = p_prev, T_prev
        else:
            print(f"[Samp ] loss={loss_samp:.2f}", end="\r")
            p_best, T_best = p_samp, T_samp

        if T_best < self.dt_control:
            self.p_next, self.T_next = None, None
            self.sol_log.append(np.vstack((q, q)))
            return p_best[-self.ndof:], np.zeros_like(dq)

        self.T_next = T_best - self.dt_control
        t_next = np.linspace(0, self.T_next, self.vptraj.N_via + 1) + self.dt_control
        q_next, dq_next, _ = self.vptraj.get_trajectory_at_time(
            t_next, p_best, q, dq0=dq, dqT=np.zeros_like(dq), T=T_best
        )
        self.p_next = q_next[1:].reshape(-1)
        self.sol_log.append(q_next)

        return q_next[0], dq_next[0]


# ========================
# Helpers
# ========================
def env_info_gen():
    robots_info = {
        "panda": {
            "urdf": PANDA_URDF,
            "base_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "conf": [-np.pi/4, 0, 0, -np.pi/2, 0, np.pi/1, np.pi/4, 0.06, 0.06],
            "scale": 1,
        },
    }
    return {"robots_info": robots_info, "utilities": {}}

def fk_xyz_for_q(pinocchio_fk: PinocchioFK, q_arm7: np.ndarray) -> np.ndarray:
    """使用Pinocchio计算给定关节的末端位置 (x,y,z)。"""
    return pinocchio_fk.compute_fk(q_arm7)


# ========================
# Main
# ========================
if __name__ == "__main__":
    # --- 创建环境 ---
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=True, mp4=None)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

    # --- 创建Pinocchio FK计算器 ---
    pinocchio_fk = PinocchioFK(PANDA_URDF)
    
    # --- 取 Panda 7 轴 ---
    arm_joints = env.get_movable_joints(env.robot, gripper=False)

    # --- 读限位 ---
    q_limits = []
    for j in arm_joints:
        low, up = get_joint_limits(env.robot, j)
        q_limits.append([low, up])
    q_limits = np.array(q_limits, dtype=float)

    # --- 速度/加速度上限 ---
    dq_limits = np.array([get_max_velocity(env.robot, j) for j in arm_joints], dtype=float)
    true_amax = np.full(7, true_amax_default, dtype=float)
    _, ddq_limits = get_dynamical_limits(env.robot, arm_joints, max_accelerations=true_amax)

    # --- 初值（使日志与仿真一致） ---
    q_full = np.array(env.get_joint_positions(env.robot), dtype=float)  # 9含夹爪
    q_arm = q_full[:7].copy()
    dq_arm = np.zeros_like(q_arm)

    # --- 计算起始和目标末端位置 ---
    start_xyz = fk_xyz_for_q(pinocchio_fk, q_arm)  # 当前位置
    goal_xyz = fk_xyz_for_q(pinocchio_fk, qg)      # 目标位置
    
    # --- 计算双平面约束 ---
    constraint_params = {"type": "none"}
    if USE_DUAL_PLANE_CONSTRAINT:
        constraint_params = compute_dual_plane_constraints(start_xyz, goal_xyz)
        print(f"[INFO] Dual plane constraint computed:")
        print(f"  - Vertical plane normal: {constraint_params.get('vertical', {}).get('normal', 'None')}")
        print(f"  - Horizontal plane normal: {constraint_params.get('horizontal', {}).get('normal', 'None')}")

    # --- 控制器 ---
    controller = DualPlaneConstraintController7D(
        N_eval=N_eval, N_via=N_via,
        vel_lim=dq_limits, acc_lim=ddq_limits,
        q_limits=q_limits, qg=qg,
        dt_control=dt_control, N_candidates=N_candidates, R_sampling=R_sampling,
        pinocchio_fk=pinocchio_fk, constraint_params=constraint_params
    )

    # --- 目标末端位姿 & 可视化 ---
    if DRAW_GOAL_AXES:
        env.render_pose(goal_xyz)  # 在目标处画坐标架
    if DRAW_GOAL_POINT:
        draw_point(goal_xyz, size=GOAL_POINT_SIZE, color=GREEN)
        
    # --- 可视化双平面约束 ---
    if USE_DUAL_PLANE_CONSTRAINT and constraint_params["type"] == "dual_plane":
        # 画起始点和目标点之间的连线
        add_line(start_xyz, goal_xyz, color=[1, 0, 0], width=3, lifetime=0.0)  # 红色连线
        draw_point(start_xyz, size=GOAL_POINT_SIZE, color=[0, 0, 1])  # 蓝色起点
        
        # 绘制双约束平面
        if DRAW_CONSTRAINT_PLANES:
            draw_dual_constraint_planes(constraint_params, 
                                      size=PLANE_SIZE, 
                                      vertical_color=VERTICAL_PLANE_COLOR,
                                      horizontal_color=HORIZONTAL_PLANE_COLOR,
                                      grid_mode=PLANE_GRID_MODE, 
                                      grid_size=PLANE_GRID_SIZE)

    # --- 分析用 ---
    q_hist = [q_arm.copy()]
    dq_hist = [dq_arm.copy()]

    # --- 末端轨迹缓存 ---
    eef_prev = None
    trail_step = 0

    # --- 主循环 ---
    total_time, acc = 0.0, 0.0
    steps = 0
    print("[INFO] Running dual plane constraint control...")
    while p.isConnected() and total_time < sim_duration:
        acc += env.dt
        total_time += env.dt

        if acc + 1e-9 >= dt_control:
            acc -= dt_control

            # 控制
            q_arm, dq_arm = controller.control(q_arm, dq_arm)
            q_hist.append(q_arm.copy())
            dq_hist.append(dq_arm.copy())

            # 下发到仿真（保持夹爪不动）
            q_full[:7] = q_arm
            q_full[-2:] = gripper_opening
            env.set_joint_positions(env.robot, q_full.tolist()[:7], gripper=False)

            # 末端轨迹（按控制周期记录，避免线段过多）
            if DRAW_EEF_TRAIL:
                trail_step += 1
                if (trail_step % TRAIL_STRIDE) == 0:
                    eef_curr = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
                    if eef_prev is not None:
                        add_line(eef_prev, eef_curr, color=GREEN, width=2, lifetime=TRAIL_LIFETIME)
                    eef_prev = eef_curr

            # 控制台进度
            if steps % int(1.0 / dt_control) == 0:
                dist = np.linalg.norm(q_arm - qg)
                
                # 计算当前双平面约束违反
                vertical_viol, horizontal_viol = 0.0, 0.0
                if USE_DUAL_PLANE_CONSTRAINT and constraint_params["type"] == "dual_plane":
                    current_eef_pos = pinocchio_fk.compute_fk(q_arm)
                    eef_pos_batch = current_eef_pos[None, None, :]  # [1, 1, 3]
                    violations = compute_dual_plane_constraint_violation(eef_pos_batch, constraint_params)
                    vertical_viol = violations["vertical"][0, 0]
                    horizontal_viol = violations["horizontal"][0, 0]
                
                if USE_DUAL_PLANE_CONSTRAINT:
                    print(f"[t={total_time:5.2f}s] dist_to_goal={dist:.3f} vertical_viol={vertical_viol:.4f} horizontal_viol={horizontal_viol:.4f}  ", end="\r")
                else:
                    print(f"[t={total_time:5.2f}s] dist_to_goal={dist:.3f}  ", end="\r")
            steps += 1

        env.step(sleep=True)

    print("\n[INFO] Done.")

    # ========== 双平面约束分析 ==========
    if USE_DUAL_PLANE_CONSTRAINT and constraint_params["type"] == "dual_plane":
        # 计算整个轨迹的双平面约束违反
        vertical_constraint_hist = []
        horizontal_constraint_hist = []
        total_constraint_hist = []
        eef_positions_hist = []
        
        print("[INFO] Computing dual plane constraint analysis...")
        for q_current in q_hist:
            eef_pos = fk_xyz_for_q(pinocchio_fk, q_current)
            eef_positions_hist.append(eef_pos)
            
            eef_pos_batch = eef_pos[None, None, :]  # [1, 1, 3]
            violations = compute_dual_plane_constraint_violation(eef_pos_batch, constraint_params)
            vertical_constraint_hist.append(violations["vertical"][0, 0])
            horizontal_constraint_hist.append(violations["horizontal"][0, 0])
            total_constraint_hist.append(violations["total"][0, 0])
        
        eef_positions_hist = np.array(eef_positions_hist)
        vertical_constraint_hist = np.array(vertical_constraint_hist)
        horizontal_constraint_hist = np.array(horizontal_constraint_hist)
        total_constraint_hist = np.array(total_constraint_hist)
        
        # 可视化双平面约束违反
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 约束违反随时间变化
        t = np.arange(len(vertical_constraint_hist)) * dt_control
        
        # 垂直平面约束违反
        ax1.plot(t, vertical_constraint_hist, 'b-', linewidth=2, label='Vertical Plane Violation')
        ax1.axhline(y=PLANE_TOLERANCE, color='r', linestyle='--', label=f'Tolerance ({PLANE_TOLERANCE}m)')
        ax1.set_ylabel('Vertical Constraint Violation (m)')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Vertical Plane Constraint Violation Over Time')
        
        # 水平平面约束违反
        ax2.plot(t, horizontal_constraint_hist, 'g-', linewidth=2, label='Horizontal Plane Violation')
        ax2.axhline(y=PLANE_TOLERANCE, color='r', linestyle='--', label=f'Tolerance ({PLANE_TOLERANCE}m)')
        ax2.set_ylabel('Horizontal Constraint Violation (m)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Horizontal Plane Constraint Violation Over Time')
        
        # 末端执行器轨迹在3D空间中的投影（XY平面）
        ax3.plot(eef_positions_hist[:, 0], eef_positions_hist[:, 1], 'b-', linewidth=2, label='EEF Trajectory')
        ax3.plot(start_xyz[0], start_xyz[1], 'go', markersize=8, label='Start')
        ax3.plot(goal_xyz[0], goal_xyz[1], 'ro', markersize=8, label='Goal')
        ax3.plot([start_xyz[0], goal_xyz[0]], [start_xyz[1], goal_xyz[1]], 'r--', linewidth=2, label='Direct Line')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('End-Effector Trajectory (XY Projection)')
        ax3.axis('equal')
        
        # 末端执行器轨迹的Z坐标变化
        ax4.plot(t, eef_positions_hist[:, 2], 'c-', linewidth=2, label='Z Position')
        ax4.axhline(y=start_xyz[2], color='g', linestyle='--', label=f'Start Z ({start_xyz[2]:.3f}m)')
        ax4.axhline(y=goal_xyz[2], color='r', linestyle='--', label=f'Goal Z ({goal_xyz[2]:.3f}m)')
        # 水平平面的Z坐标（中点）
        middle_z = (start_xyz[2] + goal_xyz[2]) / 2
        ax4.axhline(y=middle_z, color='orange', linestyle=':', label=f'Constraint Plane Z ({middle_z:.3f}m)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Z Position (m)')
        ax4.grid(True)
        ax4.legend()
        ax4.set_title('End-Effector Z Position Over Time')
        
        plt.tight_layout()
        plt.show()
        
        print(f"[INFO] Dual plane constraint analysis:")
        print(f"  - Max vertical constraint violation: {np.max(vertical_constraint_hist):.4f}m")
        print(f"  - Mean vertical constraint violation: {np.mean(vertical_constraint_hist):.4f}m")
        print(f"  - Max horizontal constraint violation: {np.max(horizontal_constraint_hist):.4f}m")
        print(f"  - Mean horizontal constraint violation: {np.mean(horizontal_constraint_hist):.4f}m")
        
        # 分析是否接近直线运动
        # 计算实际轨迹与理想直线的偏差
        trajectory_length = len(eef_positions_hist)
        ideal_trajectory = np.array([start_xyz + (goal_xyz - start_xyz) * i / (trajectory_length - 1) 
                                   for i in range(trajectory_length)])
        line_deviations = np.linalg.norm(eef_positions_hist - ideal_trajectory, axis=1)
        
        print(f"  - Max deviation from ideal straight line: {np.max(line_deviations):.4f}m")
        print(f"  - Mean deviation from ideal straight line: {np.mean(line_deviations):.4f}m")
        print(f"  - RMS deviation from ideal straight line: {np.sqrt(np.mean(line_deviations**2)):.4f}m")

    # ========== 可视化分析（关节） ==========
    fig, ax = plt.subplots(7, 1, figsize=(14, 12))
    q_hist = np.array(q_hist, dtype=float)
    t = np.arange(q_hist.shape[0]) * dt_control
    for i in range(7):
        ax[i].plot(t, q_hist[:, i], label="q")
        ax[i].hlines(q_limits[i, 0], t[0], t[-1], colors='r', linestyles='dashed', label="q_min" if i == 0 else None)
        ax[i].hlines(q_limits[i, 1], t[0], t[-1], colors='r', linestyles='dashed', label="q_max" if i == 0 else None)
        ax[i].set_ylabel(f"Joint {i+1} (rad)")
        ax[i].grid(True)
        if i == 0:
            ax[i].legend()
    ax[-1].set_xlabel("Time (s)")
    plt.suptitle("Joint States Over Time - Dual Plane Constraint")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(7, 1, figsize=(14, 12))
    dq_hist = np.array(dq_hist, dtype=float)
    t = np.arange(dq_hist.shape[0]) * dt_control
    for i in range(7):
        ax[i].plot(t, dq_hist[:, i], label="dq")
        ax[i].hlines(dq_limits[i], t[0], t[-1], colors='r', linestyles='dashed', label="dq_max" if i == 0 else None)
        ax[i].hlines(-dq_limits[i], t[0], t[-1], colors='r', linestyles='dashed', label="dq_min" if i == 0 else None)
        ax[i].set_ylabel(f"Joint {i+1} (rad/s)")
        ax[i].grid(True)
        if i == 0:
            ax[i].legend()
    ax[-1].set_xlabel("Time (s)")
    plt.suptitle("Joint Velocities Over Time - Dual Plane Constraint")
    plt.tight_layout()
    plt.show()
