"""
与 mpc_7d_panda_03.py 的区别是：
- 不使用 IK，完全靠采样自己探索 qT
- 任务空间目标可调（自己改）

从结果的简单对比，可以看出来，使用 IK 得到的 qT_hint 作为采样 bias，能更快地收敛到目标附近。
不过不使用 IK 的版本，也能收敛到目标附近，只是慢一些，而且过程中，末端位置会有较大摆动。
"""

import sys
import time
import numpy as np
import pybullet as p
from config import PANDA_URDF, PLANNER_PATH
sys.path.append(PLANNER_PATH)

from benchmarks.playground.env_simple_world import SimpleWorld
from benchmarks.utils import (
    get_joint_limits, get_max_velocity, get_dynamical_limits, GREEN,
    get_link_pose, link_from_name, get_body_name, ConfSaver,
    set_joint_positions as set_joint_positions_util, # 低层接口：可指定 joints 子集
)
from planners.vptraj import VPTraj
from matplotlib import pyplot as plt

# ========================
# 参数
# ========================
N_via = 4
N_candidates = 100
N_eval = 50
dt_control = 0.05
sim_duration = 5.0

R_sampling = 1e1           # 仅在采样时的加速度先验（平滑度），与损失分开
true_amax_default = 3.0    # rad/s^2（示意）
gripper_opening = 0.06     # 夹爪保持开口

# ========================
# 工具函数
# ========================
def env_info_gen():
    robots_info = {
        "panda": {
            "urdf": PANDA_URDF,
            "base_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "conf": [0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4, gripper_opening, gripper_opening],
            "scale": 1,
        },
    }
    return {"robots_info": robots_info, "utilities": {}}

# 末端 FK（只用到位置）
def fk_eef_pos(robot_id, tool_link, arm_joints, q_vec):
    """
    给定 7 轴关节向量 q_vec（长度7），返回末端世界坐标 (x,y,z)
    """
    saver = ConfSaver(robot_id, joints=arm_joints)
    try:
        set_joint_positions_util(robot_id, arm_joints, list(q_vec))
        pos, _ = get_link_pose(robot_id, tool_link)
        return np.array(pos, dtype=float)
    finally:
        saver.restore()

def fk_eef_pos_batch(robot_id, tool_link, arm_joints, q_batch):
    """
    q_batch: (B, 7)；返回 (B, 3)
    """
    saver = ConfSaver(robot_id, joints=arm_joints)
    out = []
    try:
        for q in q_batch:
            set_joint_positions_util(robot_id, arm_joints, list(q))
            pos, _ = get_link_pose(robot_id, tool_link)
            out.append(pos)
    finally:
        saver.restore()
    return np.asarray(out, dtype=float)

# ========================
# 控制器（任务空间末端误差，无 IK，无 qT 软约束）
# ========================
class PredictiveSamplingController7D_TS:
    def __init__(self, robot_id, tool_link, arm_joints,
                 N_eval, N_via, vel_lim, acc_lim, q_limits, xg,
                 dt_control, N_candidates, R_sampling):
        self.robot = robot_id
        self.tool_link = tool_link
        self.arm_joints = arm_joints

        self.ndof = 7
        self.vptraj = VPTraj(self.ndof, N_eval, N_via, vel_lim=np.asarray(vel_lim), acc_lim=np.asarray(acc_lim))
        self.vptraj_idle = VPTraj(self.ndof, N_eval, 1, vel_lim=np.asarray(vel_lim), acc_lim=np.asarray(acc_lim))

        self.q_limits = np.asarray(q_limits, float)  # (7,2)
        self.xg = np.asarray(xg, float)              # (3,)
        self.dt_control = float(dt_control)
        self.N_candidates = int(N_candidates)
        self.R = float(R_sampling)

        self.p_next = None
        self.T_next = None

        # 记录
        self.samples_loss_log = []
        self.sol_log = []

    def reset(self):
        self.p_next = None
        self.T_next = None
        self.samples_loss_log.clear()
        self.sol_log.clear()

    # ------- 损失函数（只看末端位置误差 + 时长 + 关节越界软惩罚）-------
    def loss_fn(self, q, dq, ddq, T):
        # q: (B, N, 7), T: (B,) or scalar
        T = np.asarray(T)
        if T.ndim == 0:
            duration_cost = np.full(q.shape[0], T, dtype=float)
        else:
            duration_cost = T.astype(float)

        # 关节越界软惩罚（只要有越界就大罚）
        low = self.q_limits[:, 0][None, None, :]
        up  = self.q_limits[:, 1][None, None, :]
        viol = (q < low) | (q > up)
        limit_violation_cost = 1e6 * np.sum(viol, axis=(1, 2))

        # 末端误差（只看终点）
        qT = q[:, -1, :]                       # (B,7)
        xT = fk_eef_pos_batch(self.robot, self.tool_link, self.arm_joints, qT)  # (B,3)
        terminal_cost = 1e3 * np.sum((xT - self.xg[None, :])**2, axis=1)

        return terminal_cost + duration_cost + limit_violation_cost

    # ------- 采样（不使用 qT 软约束，不使用 IK）-------
    def predictive_sampling(self, q, dq):
        # 注意：qT=None, Q=None，dqT=0；qT 将“包含在参数 p 里”由采样自己探索。:contentReference[oaicite:6]{index=6}
        pos, vel, acc, p, T = self.vptraj.sample_trajectories(
            self.N_candidates, q, dq0=dq, qT=None, dqT=np.zeros_like(dq), Q=None, R=self.R
        )
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], T)
        self.samples_loss_log.append(loss)
        i_best = int(np.argmin(loss))
        return p[i_best], float(loss[i_best]), float(T[i_best])

    # ------- 复用上一轮解 -------
    def previous_sol(self, q, dq):
        if self.p_next is None:
            return None, np.inf, 0.0
        pos, vel, acc = self.vptraj.get_trajectory(self.p_next, q, dq0=dq, qT=None, dqT=np.zeros_like(dq), T=self.T_next)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [self.T_next])[0]
        return self.p_next, float(loss), float(self.T_next)

    # ------- idle：尽快停下（方便在目标附近收敛/稳住）-------
    def idle(self, q, dq):
        acc_lim = self.vptraj_idle.acc_lim
        T_idle = float(np.max(np.divide(np.abs(dq), np.maximum(acc_lim, 1e-6))))
        q_idle = q + 0.5 * dq * T_idle
        pos, vel, acc = self.vptraj_idle.get_trajectory(q_idle, q, dq0=dq, qT=None, dqT=np.zeros_like(dq), T=T_idle)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [T_idle])[0]

        if T_idle < self.dt_control:
            return q_idle, np.zeros_like(dq), float(loss), 0.0

        q_next, dq_next, _ = self.vptraj_idle.get_trajectory_at_time(
            self.dt_control, q_idle, q, dq0=dq, qT=None, dqT=np.zeros_like(dq), T=T_idle
        )
        return q_next.squeeze(), dq_next.squeeze(), float(loss), T_idle

    # ------- 主控制 -------
    def control(self, q, dq):
        q_idle, dq_idle, loss_idle, T_idle = self.idle(q, dq)
        p_prev, loss_prev, T_prev = self.previous_sol(q, dq)
        p_samp, loss_samp, T_samp = self.predictive_sampling(q, dq)

        choose_idle = (loss_idle <= loss_prev) and (loss_idle <= loss_samp)
        choose_prev = (loss_prev <= loss_samp)

        if choose_idle:
            print(f"[Idle ] loss={loss_idle:.2e}", end="\r")
            self.p_next, self.T_next = None, None
            self.sol_log.append(np.vstack((q, q_idle)))
            return q_idle, dq_idle

        if choose_prev:
            print(f"[Prev ] loss={loss_prev:.2e}", end="\r")
            p_best, T_best = p_prev, T_prev
        else:
            print(f"[Samp ] loss={loss_samp:.2e}", end="\r")
            p_best, T_best = p_samp, T_samp

        if T_best < self.dt_control:
            self.p_next, self.T_next = None, None
            self.sol_log.append(np.vstack((q, q)))
            return p_best[-self.ndof:], np.zeros_like(dq)

        self.T_next = T_best - self.dt_control
        t_next = np.linspace(0, self.T_next, self.vptraj.N_via + 1) + self.dt_control
        q_next, dq_next, _ = self.vptraj.get_trajectory_at_time(
            t_next, p_best, q, dq0=dq, qT=None, dqT=np.zeros_like(dq), T=T_best
        )
        self.p_next = q_next[1:].reshape(-1)
        self.sol_log.append(q_next)
        return q_next[0], dq_next[0]


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    # --- 创建环境 ---
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=True, mp4=None)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

    # --- 取 Panda 7 轴 & 限位 ---
    arm_joints = env.get_movable_joints(env.robot, gripper=False)          # 7 轴
    q_limits = np.array([get_joint_limits(env.robot, j) for j in arm_joints], dtype=float)
    dq_limits = np.array([get_max_velocity(env.robot, j) for j in arm_joints], dtype=float)
    true_amax = np.full(7, true_amax_default, dtype=float)
    _, ddq_limits = get_dynamical_limits(env.robot, arm_joints, max_accelerations=true_amax)

    # --- 工具端 link 索引 ---
    tool_link = env.get_tool_link(env.robot)  # 'panda_grasptarget' 链接位姿作为 EEF :contentReference[oaicite:7]{index=7}

    # --- 初值（与仿真对齐）---
    q_full = np.array(env.get_joint_positions(env.robot), dtype=float)  # 长度9：7轴+2指
    q_arm = q_full[:7].copy()
    dq_arm = np.zeros_like(q_arm)

    # --- 任务空间目标（示例：从当前 EEF 再前伸 + 上抬一点）---
    q0 = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4], dtype=float)
    x_start = fk_eef_pos(env.robot, tool_link, arm_joints, q0)  # 初始位置（与仿真对齐）
    xg = np.array([0.55, 0.0, 0.90], dtype=float)

    # ---- 另一种给定方式（自己改）----
    # x_start = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)    # 当前位置
    # xg = x_start + np.array([0.15, 0.00, 0.30])                         # 目标点可自行改
    env.render_pose(xg.tolist())                                        # 可视化目标坐标系（仅位置用） :contentReference[oaicite:8]{index=8}

    # --- 控制器 ---
    controller = PredictiveSamplingController7D_TS(
        robot_id=env.robot, tool_link=tool_link, arm_joints=arm_joints,
        N_eval=N_eval, N_via=N_via,
        vel_lim=dq_limits, acc_lim=ddq_limits,
        q_limits=q_limits, xg=xg,
        dt_control=dt_control, N_candidates=N_candidates, R_sampling=R_sampling
    )

    # --- EEF 轨迹可视化 ---
    eef_prev = x_start.copy()
    trail_ids = []

    # --- 主循环 ---
    total_time, acc = 0.0, 0.0
    steps = 0
    print("[INFO] Running task-space predictive sampling (no IK, no qT-bias)...")

    eef_hist = [eef_prev.copy()]
    while p.isConnected() and total_time < sim_duration:
        acc += env.dt
        total_time += env.dt

        if acc + 1e-9 >= dt_control:
            acc -= dt_control

            q_arm, dq_arm = controller.control(q_arm, dq_arm)
            # 写回仿真（夹爪不动）
            q_full[:7] = q_arm
            q_full[-2:] = gripper_opening
            env.set_joint_positions(env.robot, q_full.tolist()[:7], gripper=False)

            # 画 EEF 轨迹线
            eef_now = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
            eef_hist.append(eef_now.copy())
            trail_ids.append(p.addUserDebugLine(eef_prev, eef_now, lineWidth=2, lineColorRGB=GREEN[:3]))
            eef_prev = eef_now.copy()

            if steps % int(max(1, 1.0/dt_control)) == 0:
                dist = np.linalg.norm(eef_now - xg)
                print(f"[t={total_time:5.2f}s] ‖eef-xg‖={dist:.4f}    ", end="\r")
            steps += 1

        env.step(sleep=True)

    print("\n[INFO] Done.")
    # 最终误差输出
    eef_final = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
    print(f"[RESULT] Final EEF dist to goal = {np.linalg.norm(eef_final - xg):.6f} m")

    # --- 末端执行器与目标的收敛情况 ---
    eef_hist = np.array(eef_hist, dtype=float)
    plt.figure()
    plt.plot(np.linalg.norm(eef_hist - xg[None, :], axis=1))
    plt.xlabel("Step")
    plt.ylabel("EEF to Goal Dist (m)")
    plt.title("End-Effector Distance to Goal Over Time")
    plt.grid()
    plt.show()