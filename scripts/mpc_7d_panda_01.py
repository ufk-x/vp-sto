"""
不考虑障碍物的 7D Panda 轨迹优化与跟踪（MPC）示例
从一个pose到另一个pose
"""

import sys
import time
import numpy as np
import pybullet as p
from config import PANDA_URDF
from config import PLANNER_PATH
sys.path.append(PLANNER_PATH)

from benchmarks.playground.env_simple_world import SimpleWorld
from benchmarks.utils import get_joint_limits, get_max_velocity, get_dynamical_limits

from planners.vptraj import VPTraj
from matplotlib import pyplot as plt

# PARAMETERS
q0 = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4], dtype=float)
qg = np.array([0, -np.pi/4, 0, -np.pi/2, 0, np.pi/2, np.pi/4], dtype=float)
dq0 = np.zeros(7, dtype=float)
dqg = np.zeros(7, dtype=float)

# 采样/代价参数
R_sampling = 1e1         # 采样阶段的最小加速度能量惩罚（标量！）
Q_min = 1e0              # 末端误差精度（下限）
Q_max = 1e3              # 末端误差精度（上限）
factor_Q_min = 1e-1      # Q自适应缩放因子下限
factor_Q_max = 1e1       # Q自适应缩放因子上限

N_via = 4                # via-points 个数
N_candidates = 100       # 每轮采样候选数量
N_eval = 50              # 轨迹离散点数（用于代价评估）
dt_control = 0.05        # 控制周期（s）
sim_duration = 20.0      # 总仿真时长（s）

dt_control = 0.05     # Time step for control (s)
sim_duration = 5     # Duration of simulation (s)

# 若无真实 a_max，可用启发式。这里给个示例真值（可按需求改）
true_amax_default = 3.0  # rad/s^2（示意）
gripper_opening = 0.06   # 夹爪保持开口（不参与7D控制）


# ========================
# Controller
# ========================
class PredictiveSamplingController7D:
    def __init__(self, N_eval, N_via, vel_lim, acc_lim, q_limits, qg,
                 dt_control, N_candidates, R_sampling):
        self.ndof = 7
        self.vptraj = VPTraj(ndof=self.ndof, N_eval=N_eval, N_via=N_via,
                             vel_lim=np.asarray(vel_lim, dtype=float),
                             acc_lim=np.asarray(acc_lim, dtype=float))  # 限制在构造时生效  :contentReference[oaicite:3]{index=3}
        # 一个“idle”轨迹发生器（via=1，快停）
        self.vptraj_idle = VPTraj(ndof=self.ndof, N_eval=N_eval, N_via=1,
                                  vel_lim=np.asarray(vel_lim, dtype=float),
                                  acc_lim=np.asarray(acc_lim, dtype=float))

        self.q_limits = np.asarray(q_limits, dtype=float)  # 形状：(7, 2)
        self.qg = np.asarray(qg, dtype=float)  # shape: (7,)
        self.dt_control = float(dt_control)
        self.N_candidates = int(N_candidates)
        self.R = float(R_sampling)

        # 自适应 Q（目标点吸引强度）
        self.Q = Q_max
        self.Q_log = []

        # 复用解
        self.p_next = None
        self.T_next = None

        # 可视化日志
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

    # --------- 损失函数 ----------
    def loss_fn(self, q, dq, ddq, T):
        """
        q: (B, N, 7), dq/ddq 同维；T: (B,) 或 标量
        """
        # 时间代价（鼓励更短 T）
        T = np.asarray(T)
        if T.ndim == 0:
            duration_cost = np.full(q.shape[0], T)
        else:
            duration_cost = T

        # 终端误差
        qT = q[:, -1, :]  # 末端关节
        terminal_cost = 1e3 * np.sum((qT - self.qg)**2, axis=1)

        # 关节位置越界软约束
        low = self.q_limits[:, 0][None, None, :]  # (1,1,7)
        up  = self.q_limits[:, 1][None, None, :]
        viol = (q < low) | (q > up)
        limit_violation_cost = 1e6 * np.sum(viol, axis=(1, 2))

        # 合成
        return terminal_cost + duration_cost + limit_violation_cost
    
    # --------- 采样策略 ----------
    def predictive_sampling(self, q, dq):
        # 基于上一轮违反约束的比例，自适应调节 Q
        if len(self.samples_loss_log) > 0:
            num_viol = np.sum(self.samples_loss_log[-1] > 1e6)
        else:
            num_viol = 0
        ratio = num_viol / max(1, self.N_candidates)
        self.Q *= np.clip(np.exp(-3 * (ratio - 0.5)), factor_Q_min, factor_Q_max)
        self.Q = np.clip(self.Q, Q_min, Q_max)
        self.Q_log.append(self.Q)

        # 采样候选轨迹并评估
        pos, vel, acc, p, T = self.vptraj.sample_trajectories(
            self.N_candidates, q, dq0=dq, qT=self.qg, dqT=np.zeros_like(dq),
            Q=self.Q, R=self.R
        )  # 内部会根据 vel/acc 上限自动求最小 T 以满足约束  :contentReference[oaicite:4]{index=4}

        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], T)
        self.samples_log.append(pos)
        self.samples_loss_log.append(loss)
        i_best = int(np.argmin(loss))
        return p[i_best], loss[i_best], float(T[i_best])

    # --------- 复用上一轮解 ----------
    def previous_sol(self, q, dq):
        if self.p_next is None:
            return None, np.inf, 0.0
        pos, vel, acc = self.vptraj.get_trajectory(self.p_next, q, dq0=dq,
                                                   dqT=np.zeros_like(dq), T=self.T_next)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [self.T_next])[0]
        return self.p_next, loss, self.T_next
    
    # --------- idle：尽快停下 ----------
    def idle(self, q, dq):
        # 常加速度模型：T_idle = max_i |dq_i| / a_max_i
        acc_lim = self.vptraj_idle.acc_lim
        T_idle = float(np.max(np.divide(np.abs(dq), np.maximum(acc_lim, 1e-6))))
        # 停止位置的近似（s = v*T/2）
        q_idle = q + 0.5 * dq * T_idle
        pos, vel, acc = self.vptraj_idle.get_trajectory(q_idle, q, dq0=dq,
                                                        dqT=np.zeros_like(dq), T=T_idle)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [T_idle])[0]

        # 如果很快就停完，直接落地到 q_idle
        if T_idle < self.dt_control:
            return q_idle, np.zeros_like(dq), loss, 0.0

        # 否则取 dt_control 后的点作为下一步参考
        q_next, dq_next, _ = self.vptraj_idle.get_trajectory_at_time(
            self.dt_control, q_idle, q, dq0=dq, dqT=np.zeros_like(dq), T=T_idle
        )
        return q_next.squeeze(), dq_next.squeeze(), loss, T_idle
    
    # --------- 主控制 ----------
    def control(self, q, dq):
        # 三种策略：idle / reuse / sampling
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
            # 轨迹很短，直接收尾：把 p 的末尾（dqT）当作下一个姿态增量点
            self.p_next, self.T_next = None, None
            self.sol_log.append(np.vstack((q, q)))
            return p_best[-self.ndof:], np.zeros_like(dq)

        # 递推到下一拍：把 (t in [dt, T_best]) 的采样作为下一轮的 via 参数
        self.T_next = T_best - self.dt_control
        t_next = np.linspace(0, self.T_next, self.vptraj.N_via + 1) + self.dt_control
        q_next, dq_next, _ = self.vptraj.get_trajectory_at_time(
            t_next, p_best, q, dq0=dq, dqT=np.zeros_like(dq), T=T_best
        )
        self.p_next = q_next[1:].reshape(-1)  # via 参数
        self.sol_log.append(q_next)

        return q_next[0], dq_next[0]


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


if __name__ == "__main__":
    # --- 创建环境 ---
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=True, mp4=None)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

    # --- 取 Panda 7 轴 ---
    arm_joints = env.get_movable_joints(env.robot, gripper=False)  # 7关节索引（不含夹爪）

    # --- 读限位 ---
    q_limits = []
    for j in arm_joints:
        low, up = get_joint_limits(env.robot, j)  # 单关节上下限
        q_limits.append([low, up])
    q_limits = np.array(q_limits, dtype=float)

    # --- 速度/加速度上限 ---
    dq_limits = []
    for j in arm_joints:
        dq_limits.append(get_max_velocity(env.robot, j))  # 单关节 v_max
    dq_limits = np.array(dq_limits, dtype=float)

    # 若你有真实 a_max，直接传给 get_dynamical_limits；否则它会用启发式 duration_to_max 估计
    true_amax = np.full(7, true_amax_default, dtype=float)
    _, ddq_limits = get_dynamical_limits(env.robot, arm_joints, max_accelerations=true_amax)

    # --- 控制器 ---
    controller = PredictiveSamplingController7D(
        N_eval=N_eval, N_via=N_via,
        vel_lim=dq_limits, acc_lim=ddq_limits,
        q_limits=q_limits, qg=qg,
        dt_control=dt_control, N_candidates=N_candidates, R_sampling=R_sampling
    )

    # --- 初值（以环境实测为准，确保与仿真一致） ---
    q_full = np.array(env.get_joint_positions(env.robot), dtype=float)  # 长度=9（含夹爪）
    q_arm = q_full[:7].copy()
    dq_arm = np.zeros_like(q_arm)

    # 若想强制用 q0 起步，放开下面两行
    # q_arm = q0.copy()
    # env.set_joint_positions(env.robot, list(q_arm) + [gripper_opening, gripper_opening], gripper=True)

    # --- 分析用 ---
    q_hist = [q_arm.copy()]
    dq_hist = [dq_arm.copy()]

    # --- 主循环 ---
    total_time, acc = 0.0, 0.0
    steps = 0
    print("[INFO] Running predictive sampling control...")
    while p.isConnected() and total_time < sim_duration:
        # 控制频率：每 dt_control 触发一次优化
        acc += env.dt
        total_time += env.dt

        if acc + 1e-9 >= dt_control:
            acc -= dt_control

            # 调用控制器
            q_arm, dq_arm = controller.control(q_arm, dq_arm)
            q_hist.append(q_arm.copy())
            dq_hist.append(dq_arm.copy())

            # 下发到仿真（保持夹爪不动）
            q_full[:7] = q_arm
            q_full[-2:] = gripper_opening
            env.set_joint_positions(env.robot, q_full.tolist(), gripper=True)

            # 控制台进度
            if steps % int(1.0 / dt_control) == 0:
                dist = np.linalg.norm(q_arm - qg)
                print(f"[t={total_time:5.2f}s] dist_to_goal={dist:.3f}  ", end="\r")
            steps += 1

        env.step(sleep=True)

    print("\n[INFO] Done.")

    # --- 可视化分析 ---
    # 关节转角分析
    fig, ax = plt.subplots(7, 1, figsize=(14, 12))
    q_hist = np.array(q_hist, dtype=float)
    t = np.arange(q_hist.shape[0]) * dt_control
    for i in range(7):
        # 画出关节角度
        ax[i].plot(t, q_hist[:, i], label="q")
        # 画出关节角度约束
        ax[i].hlines(q_limits[i, 0], t[0], t[-1], colors='r', linestyles='dashed', label="q_min" if i == 0 else None)
        ax[i].hlines(q_limits[i, 1], t[0], t[-1], colors='r', linestyles='dashed', label="q_max" if i == 0 else None)
        ax[i].set_ylabel(f"Joint {i+1} (rad/rad/s)")
        ax[i].grid(True)
        if i == 0:
            ax[i].legend()
    ax[-1].set_xlabel("Time (s)")
    plt.suptitle("Joint States Over Time")
    plt.tight_layout()
    plt.show()

    # 关节角速度分析
    fig, ax = plt.subplots(7, 1, figsize=(14, 12))
    dq_hist = np.array(dq_hist, dtype=float)
    t = np.arange(dq_hist.shape[0]) * dt_control
    for i in range(7):
        # 画出关节角速度
        ax[i].plot(t, dq_hist[:, i], label="dq")
        # 画出关节速度约束
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

