"""
panda_02脚本在01的基础上，增加了对末端轨迹的可视化。调用了一些其他接口。
"""

import sys
import time
import numpy as np
import pybullet as p
import pinocchio as pin
from config import PANDA_URDF
from config import PLANNER_PATH
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
DRAW_CONSTRAINT_PLANE = True  # 是否画约束平面
PLANE_SIZE = 0.5              # 平面半尺寸 (m)
PLANE_ALPHA = 0.3             # 平面透明度
PLANE_COLOR = [0.8, 0.8, 0.2] # 平面颜色 (黄色)
PLANE_GRID_MODE = True        # 是否使用网格模式（更清晰）
PLANE_GRID_SIZE = 8           # 网格密度

LOSS_GOAL = 1e3
# ===== 路径约束参数 =====
USE_PATH_CONSTRAINT = True    # 是否启用路径约束
CONSTRAINT_TYPE = "plane"     # "plane" 或 "line"
PATH_CONSTRAINT_WEIGHT = 1e6  # 路径约束惩罚权重
PLANE_TOLERANCE = 0.01        # 平面约束容忍度 (m)
LINE_TOLERANCE = 0.02         # 直线约束容忍度 (m)


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
# Path Constraint Functions
# ========================
def compute_path_constraints(start_pos, goal_pos, constraint_type="plane"):
    """
    计算路径约束参数
    
    Args:
        start_pos: 起始位置 [x, y, z]
        goal_pos: 目标位置 [x, y, z]
        constraint_type: "plane" 或 "line"
    
    Returns:
        constraint_params: 约束参数字典
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
    
    if constraint_type == "line":
        # 直线约束：点到直线距离
        return {
            "type": "line",
            "start_pos": start_pos,
            "direction": direction_unit,
            "length": length
        }
    
    elif constraint_type == "plane":
        # 平面约束：包含连线且平行于z轴的平面
        # 平面法向量：连线在xy平面的投影的垂直方向
        direction_xy = direction[:2]  # [dx, dy]
        length_xy = np.linalg.norm(direction_xy)
        
        if length_xy < 1e-6:
            # 连线完全垂直，无约束
            return {"type": "none"}
        
        # 平面法向量在xy平面内，垂直于连线投影
        normal_xy = np.array([-direction_xy[1], direction_xy[0]]) / length_xy
        normal = np.array([normal_xy[0], normal_xy[1], 0.0])  # z分量为0
        
        # 平面方程：normal · (P - start_pos) = 0
        return {
            "type": "plane",
            "normal": normal,
            "point": start_pos
        }
    
    return {"type": "none"}

def compute_path_constraint_violation(positions, constraint_params):
    """
    计算路径约束违反程度
    
    Args:
        positions: 末端执行器位置数组，形状 [N_candidates, N_timesteps, 3]
        constraint_params: 约束参数
    
    Returns:
        violations: 违反程度数组 [N_candidates, N_timesteps]
    """
    if constraint_params["type"] == "none":
        return np.zeros(positions.shape[:2])
    
    elif constraint_params["type"] == "line":
        # 点到直线距离
        start_pos = constraint_params["start_pos"]
        direction = constraint_params["direction"]
        
        # 计算每个点到直线的距离
        # P为轨迹点，A为直线起点，d为方向向量
        # 距离 = ||(P-A) - ((P-A)·d)d||
        AP = positions - start_pos[None, None, :]  # [N_candidates, N_timesteps, 3]
        proj_length = np.sum(AP * direction[None, None, :], axis=2, keepdims=True)  # 投影长度
        proj_vector = proj_length * direction[None, None, :]  # 投影向量
        perpendicular = AP - proj_vector  # 垂直分量
        distances = np.linalg.norm(perpendicular, axis=2)  # [N_candidates, N_timesteps]
        
        return distances
    
    elif constraint_params["type"] == "plane":
        # 点到平面距离
        normal = constraint_params["normal"]
        point = constraint_params["point"]
        
        # 平面方程：normal · (P - point) = 0
        # 距离 = |normal · (P - point)| / ||normal||
        PP0 = positions - point[None, None, :]  # [N_candidates, N_timesteps, 3]
        distances = np.abs(np.sum(PP0 * normal[None, None, :], axis=2))  # [N_candidates, N_timesteps]
        
        return distances
    
    return np.zeros(positions.shape[:2])


# ========================
# Visualization Helpers
# ========================
def draw_constraint_plane(start_pos, goal_pos, constraint_params, size=0.5, color=[0.8, 0.8, 0.2], 
                        grid_mode=True, grid_size=8):
    """
    在PyBullet中绘制约束平面
    
    Args:
        start_pos: 起始位置 [x, y, z]
        goal_pos: 目标位置 [x, y, z]
        constraint_params: 约束参数
        size: 平面半尺寸
        color: 平面颜色
        grid_mode: 是否使用网格模式
        grid_size: 网格密度
    """
    if constraint_params["type"] != "plane":
        return None
    
    normal = constraint_params["normal"]  # 平面法向量
    point = constraint_params["point"]    # 平面上一点
    
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
    v_direction = np.array([0, 0, 1])  # z方向
    
    # 如果连线已经是z方向，选择另一个方向
    if abs(np.dot(u_direction, v_direction)) > 0.99:
        v_direction = np.array([1, 0, 0])  # x方向
    
    if grid_mode:
        # 网格模式：绘制详细网格
        step_u = 2 * size / grid_size
        step_v = 2 * size / grid_size
        line_color = [c * 0.8 for c in color]
        
        # U方向的网格线（平行于连线）
        for i in range(grid_size + 1):
            v_offset = -size + i * step_v
            start_grid = center_on_plane - size * u_direction + v_offset * v_direction
            end_grid = center_on_plane + size * u_direction + v_offset * v_direction
            width = 2 if i == 0 or i == grid_size or i == grid_size//2 else 1
            add_line(start_grid, end_grid, color=line_color, width=width, lifetime=0.0)
        
        # V方向的网格线（垂直于连线）
        for i in range(grid_size + 1):
            u_offset = -size + i * step_u
            start_grid = center_on_plane + u_offset * u_direction - size * v_direction
            end_grid = center_on_plane + u_offset * u_direction + size * v_direction
            width = 2 if i == 0 or i == grid_size or i == grid_size//2 else 1
            add_line(start_grid, end_grid, color=line_color, width=width, lifetime=0.0)
    else:
        # 简单模式：只绘制基本结构
        corners = [
            center_on_plane - size * u_direction - size * v_direction,
            center_on_plane + size * u_direction - size * v_direction,
            center_on_plane + size * u_direction + size * v_direction,
            center_on_plane - size * u_direction + size * v_direction
        ]
        
        line_color = [c * 0.7 for c in color]
        # 绘制对角线
        add_line(corners[0], corners[2], color=line_color, width=1, lifetime=0.0)
        add_line(corners[1], corners[3], color=line_color, width=1, lifetime=0.0)
    
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
    
    # 绘制连线在平面上的投影（用不同颜色区分）
    start_proj = start_pos - np.dot(normal, start_pos - point) * normal
    goal_proj = goal_pos - np.dot(normal, goal_pos - point) * normal
    add_line(start_proj, goal_proj, color=[0, 1, 0], width=4, lifetime=0.0)  # 绿色投影线
    
    print(f"[INFO] Constraint plane visualized:")
    print(f"  - Center: {center_on_plane}")
    print(f"  - Normal: {normal}")
    print(f"  - Size: {size}m x {size}m")
    print(f"  - Mode: {'Grid' if grid_mode else 'Simple'}")
    if grid_mode:
        print(f"  - Grid: {grid_size}x{grid_size}")
    
    return center_on_plane


# ========================
# Controller（原样）
# ========================
class PredictiveSamplingController7D:
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
        
        # 路径约束相关 - 使用Pinocchio FK
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

        # 路径约束惩罚
        path_constraint_cost = 0.0
        if USE_PATH_CONSTRAINT and self.pinocchio_fk is not None and self.constraint_params["type"] != "none":
            # 计算轨迹上每个点的末端执行器位置
            eef_positions = self._compute_eef_positions_batch(q)  # [N_candidates, N_timesteps, 3]
            
            # 计算约束违反
            violations = compute_path_constraint_violation(eef_positions, self.constraint_params)
            
            # 计算惩罚
            if self.constraint_params["type"] == "line":
                tolerance = LINE_TOLERANCE
            else:  # plane
                tolerance = PLANE_TOLERANCE
            
            # 超出容忍度的部分进行惩罚
            excess_violations = np.maximum(0, violations - tolerance)
            path_constraint_cost = PATH_CONSTRAINT_WEIGHT * np.sum(excess_violations**2, axis=1)

        return terminal_cost + duration_cost + limit_violation_cost + path_constraint_cost
    
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
    
    # --- 计算路径约束 ---
    constraint_params = {"type": "none"}
    if USE_PATH_CONSTRAINT:
        constraint_params = compute_path_constraints(start_xyz, goal_xyz, CONSTRAINT_TYPE)
        print(f"[INFO] Path constraint: {constraint_params['type']}")
        if constraint_params["type"] == "plane":
            print(f"[INFO] Plane normal: {constraint_params['normal']}")
        elif constraint_params["type"] == "line":
            print(f"[INFO] Line direction: {constraint_params['direction']}")

    # --- 控制器 ---
    controller = PredictiveSamplingController7D(
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
        
    # --- 可视化路径约束 ---
    if USE_PATH_CONSTRAINT and constraint_params["type"] != "none":
        # 画起始点和目标点之间的连线
        add_line(start_xyz, goal_xyz, color=[1, 0, 0], width=3, lifetime=0.0)  # 红色连线
        draw_point(start_xyz, size=GOAL_POINT_SIZE, color=[0, 0, 1])  # 蓝色起点
        
        # 绘制约束平面
        if DRAW_CONSTRAINT_PLANE and constraint_params["type"] == "plane":
            draw_constraint_plane(start_xyz, goal_xyz, constraint_params, 
                                size=PLANE_SIZE, color=PLANE_COLOR, 
                                grid_mode=PLANE_GRID_MODE, grid_size=PLANE_GRID_SIZE)

    # --- 分析用 ---
    q_hist = [q_arm.copy()]
    dq_hist = [dq_arm.copy()]

    # --- 末端轨迹缓存 ---
    eef_prev = None
    trail_step = 0

    # --- 主循环 ---
    total_time, acc = 0.0, 0.0
    steps = 0
    print("[INFO] Running predictive sampling control...")
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
            env.set_joint_positions(env.robot, q_full.tolist()[:7], gripper=False) # 做了修改，防止夹爪可视化抖动

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
                
                # 计算当前路径约束违反
                constraint_viol = 0.0
                if USE_PATH_CONSTRAINT and constraint_params["type"] != "none":
                    current_eef_pos = pinocchio_fk.compute_fk(q_arm)
                    eef_pos_batch = current_eef_pos[None, None, :]  # [1, 1, 3]
                    viol = compute_path_constraint_violation(eef_pos_batch, constraint_params)
                    constraint_viol = viol[0, 0]
                
                if USE_PATH_CONSTRAINT:
                    print(f"[t={total_time:5.2f}s] dist_to_goal={dist:.3f} constraint_viol={constraint_viol:.4f}  ", end="\r")
                else:
                    print(f"[t={total_time:5.2f}s] dist_to_goal={dist:.3f}  ", end="\r")
            steps += 1

        env.step(sleep=True)

    print("\n[INFO] Done.")

    # ========== 路径约束分析 ==========
    if USE_PATH_CONSTRAINT and constraint_params["type"] != "none":
        # 计算整个轨迹的路径约束违反
        constraint_hist = []
        eef_positions_hist = []
        
        print("[INFO] Computing path constraint analysis...")
        for q_current in q_hist:
            eef_pos = fk_xyz_for_q(pinocchio_fk, q_current)
            eef_positions_hist.append(eef_pos)
            
            eef_pos_batch = eef_pos[None, None, :]  # [1, 1, 3]
            viol = compute_path_constraint_violation(eef_pos_batch, constraint_params)
            constraint_hist.append(viol[0, 0])
        
        eef_positions_hist = np.array(eef_positions_hist)
        constraint_hist = np.array(constraint_hist)
        
        # 可视化路径约束违反
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 约束违反随时间变化
        t = np.arange(len(constraint_hist)) * dt_control
        ax1.plot(t, constraint_hist, 'b-', linewidth=2, label='Constraint Violation')
        if constraint_params["type"] == "line":
            ax1.axhline(y=LINE_TOLERANCE, color='r', linestyle='--', label=f'Tolerance ({LINE_TOLERANCE}m)')
        else:
            ax1.axhline(y=PLANE_TOLERANCE, color='r', linestyle='--', label=f'Tolerance ({PLANE_TOLERANCE}m)')
        ax1.set_ylabel('Constraint Violation (m)')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title(f'Path Constraint Violation Over Time ({constraint_params["type"].upper()})')
        
        # 末端执行器轨迹在3D空间中的投影
        ax2.plot(eef_positions_hist[:, 0], eef_positions_hist[:, 1], 'b-', linewidth=2, label='EEF Trajectory')
        ax2.plot(start_xyz[0], start_xyz[1], 'go', markersize=8, label='Start')
        ax2.plot(goal_xyz[0], goal_xyz[1], 'ro', markersize=8, label='Goal')
        ax2.plot([start_xyz[0], goal_xyz[0]], [start_xyz[1], goal_xyz[1]], 'r--', linewidth=2, label='Direct Line')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('End-Effector Trajectory (XY Projection)')
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"[INFO] Max constraint violation: {np.max(constraint_hist):.4f}m")
        print(f"[INFO] Mean constraint violation: {np.mean(constraint_hist):.4f}m")

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
    plt.suptitle("Joint States Over Time")
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
    plt.suptitle("Joint Velocities Over Time")
    plt.tight_layout()
    plt.show()
