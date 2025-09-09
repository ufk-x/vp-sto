"""
panda_02脚本在01的基础上，增加了对末端轨迹的可视化。调用了一些其他接口。
"""

import sys
import time
import numpy as np
import pybullet as p
from config import PANDA_URDF
from config import PLANNER_PATH
sys.path.append(PLANNER_PATH)

from benchmarks.playground.env_simple_world import SimpleWorld
from benchmarks.utils import (
    get_joint_limits, get_max_velocity, get_dynamical_limits,
    # 下面这些是为 FK&可视化新增：
    ConfSaver, set_joint_positions, get_link_pose,   # FK 用
    draw_point, add_line, GREEN, remove_debug,       # 可视化用
    Pose, draw_pose                                  # 末端坐标轴用
)

from planners.vptraj import VPTraj
from matplotlib import pyplot as plt

# ========================
# PARAMETERS
# ========================
q0 = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4], dtype=float)
qg = np.array([0, -np.pi/4, 0, -np.pi/2, 0, np.pi/2, np.pi/4], dtype=float)
dq0 = np.zeros(7, dtype=float)
dqg = np.zeros(7, dtype=float)

# 采样/代价参数
R_sampling = 1e1
Q_min = 1e0
Q_max = 1e5
factor_Q_min = 1e-1
factor_Q_max = 1e1

N_via = 4
N_candidates = 100
N_eval = 50
dt_control = 0.05        # 控制周期（s）
sim_duration = 15.0       # 仿真总时长（s）

true_amax_default = 3.0  # rad/s^2（示意）
gripper_opening = 0.06   # 夹爪开口

# ===== 可视化参数（你可以随时改） =====
DRAW_GOAL_AXES = True         # 是否画目标位姿坐标架
DRAW_GOAL_POINT = True        # 是否画目标位置点标
GOAL_POINT_SIZE = 0.02
DRAW_EEF_AXES = True          # 是否画末端位姿坐标架
EEF_AXES_WIDTH = 15            # 末端坐标架线宽
DRAW_EEF_TRAIL = True         # 是否画末端轨迹
TRAIL_LIFETIME = 0.0          # 线段寿命（秒），0=永久
TRAIL_STRIDE = 1              # 每多少个控制周期画一次线（>=1）


# ========================
# Controller（原样）
# ========================
class PredictiveSamplingController7D:
    def __init__(self, N_eval, N_via, vel_lim, acc_lim, q_limits, qg,
                 dt_control, N_candidates, R_sampling):
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
        terminal_cost = 1e3 * np.sum((qT - self.qg)**2, axis=1)
        low = self.q_limits[:, 0][None, None, :]
        up  = self.q_limits[:, 1][None, None, :]
        viol = (q < low) | (q > up)
        limit_violation_cost = 1e6 * np.sum(viol, axis=(1, 2))
        return terminal_cost + duration_cost + limit_violation_cost

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
            "conf": [0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4, 0.06, 0.06],
            "scale": 1,
        },
    }
    return {"robots_info": robots_info, "utilities": {}}

def fk_xyz_for_q(env: SimpleWorld, q_arm7: np.ndarray) -> np.ndarray:
    """不改变仿真状态，离线计算给定关节的末端位置 (x,y,z)。"""
    tool_link = env.get_tool_link(env.robot)  # panda_grasptarget
    arm_joints = env.get_movable_joints(env.robot, gripper=False)
    with ConfSaver(env.robot, joints=arm_joints):
        set_joint_positions(env.robot, arm_joints, q_arm7)
        pos, _ = get_link_pose(env.robot, tool_link)  # worldLinkFramePosition
    return np.array(pos, dtype=float)


# ========================
# Main
# ========================
if __name__ == "__main__":
    # --- 创建环境 ---
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=True, mp4=None)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

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

    # --- 控制器 ---
    controller = PredictiveSamplingController7D(
        N_eval=N_eval, N_via=N_via,
        vel_lim=dq_limits, acc_lim=ddq_limits,
        q_limits=q_limits, qg=qg,
        dt_control=dt_control, N_candidates=N_candidates, R_sampling=R_sampling
    )

    # --- 初值（使日志与仿真一致） ---
    q_full = np.array(env.get_joint_positions(env.robot), dtype=float)  # 9含夹爪
    q_arm = q_full[:7].copy()
    dq_arm = np.zeros_like(q_arm)

    # --- 目标末端位姿 & 可视化 ---
    goal_xyz = fk_xyz_for_q(env, qg)  # 给定 qg 的末端位置
    if DRAW_GOAL_AXES:
        env.render_pose(goal_xyz, width=15)  # 在目标处画坐标架，使用更粗的线条
    if DRAW_GOAL_POINT:
        draw_point(goal_xyz, size=GOAL_POINT_SIZE, color=GREEN)

    # --- 分析用 ---
    q_hist = [q_arm.copy()]
    dq_hist = [dq_arm.copy()]

    # --- 末端轨迹缓存 ---
    eef_prev = None
    trail_step = 0
    eef_axes_handles = []  # 存储末端坐标轴的句柄，用于清除旧的坐标轴

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

            # 绘制末端坐标轴
            if DRAW_EEF_AXES:
                # 清除旧的坐标轴
                for handle in eef_axes_handles:
                    remove_debug(handle)
                eef_axes_handles.clear()
                
                # 获取当前末端位置和姿态
                eef_pose_list = env.get_eef_pose(env.robot)
                eef_point = eef_pose_list[:3]  # [x, y, z]
                eef_euler = eef_pose_list[3:]  # [roll, pitch, yaw]
                
                # 创建 Pose 对象并绘制坐标轴
                pose = Pose(eef_point, eef_euler)
                handles = draw_pose(pose, length=0.08, width=EEF_AXES_WIDTH)
                eef_axes_handles.extend(handles)

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
                print(f"[t={total_time:5.2f}s] dist_to_goal={dist:.3f}  ", end="\r")
            steps += 1

        env.step(sleep=True)

    print("\n[INFO] Done.")

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
