"""
与 mpc_7d_panda_05.py 的区别是：
- 不使用 IK，完全靠采样自己探索 qT
- 任务空间目标可调（自己改）
- 用一组“跟随 link 的小球”可视化 Panda 的几何近似（collision spheres）
- 新增：2-3 个动态障碍物小球 + 可视化 + 避障代价（终点基于 FK 的距离惩罚）
- 这种避障是非常保守的，因为只考虑了机器人轨迹的最后一帧（终点），且障碍物位置是基于当前帧估计的。
- 这个版本的FK主要是基于pybullet的，可视化会有闪现问题。
"""

import sys
import time
import yaml
import numpy as np
import pybullet as p
from config import PANDA_URDF, PLANNER_PATH, COLLISION_MODELS_PATH
sys.path.append(PLANNER_PATH)

from benchmarks.playground.env_simple_world import SimpleWorld
from benchmarks.utils import (
    get_joint_limits, get_max_velocity, get_dynamical_limits, GREEN,
    get_link_pose, link_from_name, ConfSaver,
    set_joint_positions as set_joint_positions_util,
    create_sphere, STATIC_MASS,
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
sim_duration = 8.0

R_sampling = 1e1           # 采样平滑先验
true_amax_default = 3.0    # rad/s^2
gripper_opening = 0.06

# 避障代价参数
LAMBDA_AVOID   = 1.0e4     # 避障权重（可加大至 1e4 强化）
SAFETY_MARGIN  = 0.03      # 额外安全距离（米）

# 可视化
DRAW_CLOSEST_PAIR = True   # 是否画最近“机器人球-障碍球”连线
CLOSEST_LINE_LIFE = 0.15   # 连线显示寿命（秒）

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

# ========================
# 末端 FK（只用到位置）
# ========================
def fk_eef_pos_batch(robot_id, tool_link, arm_joints, q_batch):
    """q_batch: (B, 7)；返回 (B, 3)"""
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
# 机器人小球模型（计算任意 q 下所有球的世界坐标）
# ========================
class RobotSpheresModel:
    def __init__(self, robot, arm_joints, spec_dict):
        """
        spec_dict: {'link_name': [{'center':[x,y,z], 'radius':r}, ...], ...}
        """
        self.robot = robot
        self.arm_joints = arm_joints

        self.link_indices = []
        self.local_centers = []  # list of (Ni,3)
        self.local_radii   = []  # list of (Ni,)
        for link_name, spheres in spec_dict.items():
            link = link_from_name(robot, link_name)
            self.link_indices.append(link)
            centers = np.array([s["center"] for s in spheres], dtype=float)
            radii   = np.array([s["radius"] for s in spheres], dtype=float)
            self.local_centers.append(centers)
            self.local_radii.append(radii)

        self.num_links = len(self.link_indices)
        self.num_spheres = int(sum(len(r) for r in self.local_radii))

    def world_spheres_for_q(self, q):
        """
        给定 7 轴关节 q，返回：
          centers_world: (M,3)
          radii:         (M,)
        """
        saver = ConfSaver(self.robot, joints=self.arm_joints)
        centers_world = []
        radii_world   = []
        try:
            set_joint_positions_util(self.robot, self.arm_joints, list(q))
            for link, C_local, R_local in zip(self.link_indices, self.local_centers, self.local_radii):
                link_pos, link_quat = get_link_pose(self.robot, link)
                for c, r in zip(C_local, R_local):
                    wp, wq = p.multiplyTransforms(link_pos, link_quat, c.tolist(), [0,0,0,1])
                    centers_world.append(wp); radii_world.append(r)
        finally:
            saver.restore()
        return np.array(centers_world, dtype=float), np.array(radii_world, dtype=float)

    def batch_world_spheres_for_qT(self, qT_batch):
        """
        对一批终点 qT_batch: (B,7) 逐个计算机器人球的世界坐标。
        返回：
          list_centers: 长度 B，每项 (M,3)
          list_radii:   长度 B，每项 (M,)
        """
        list_centers, list_radii = [], []
        for q in qT_batch:
            C, R = self.world_spheres_for_q(q)
            list_centers.append(C); list_radii.append(R)
        return list_centers, list_radii

# ========================
# 可视化：跟随 link 的小球（仅显示用途）
# ========================
class CollisionSpheresVis:
    def __init__(self, robot, color=(0.2, 0.6, 1.0, 0.6)):
        self.robot = robot
        self.color = color
        self._items = []  # 每项：{link, local_centers(N,3), radii(N,), sphere_ids(N,)}

    def add_from_spec(self, spec: dict):
        for link_name, spheres in spec.items():
            link = link_from_name(self.robot, link_name)
            local_centers, radii, ids = [], [], []
            for s in spheres:
                c = np.asarray(s["center"], dtype=float)
                r = float(s["radius"])
                sid = create_sphere(r, mass=STATIC_MASS, color=self.color)
                # 关闭碰撞 & 干扰
                p.setCollisionFilterGroupMask(sid, -1, 0, 0)
                p.changeDynamics(sid, -1, lateralFriction=0, spinningFriction=0,
                                 rollingFriction=0, linearDamping=0, angularDamping=0)
                local_centers.append(c); radii.append(r); ids.append(sid)
            self._items.append(dict(
                link=link,
                local_centers=np.vstack(local_centers),
                radii=np.asarray(radii, dtype=float),
                sphere_ids=np.asarray(ids, dtype=int),
            ))

    def update(self):
        for item in self._items:
            link = item["link"]
            centers = item["local_centers"]
            sphere_ids = item["sphere_ids"]
            link_pos, link_quat = get_link_pose(self.robot, link)
            for c_local, sid in zip(centers, sphere_ids):
                world_pos, world_quat = p.multiplyTransforms(
                    link_pos, link_quat, c_local.tolist(), [0, 0, 0, 1]
                )
                p.resetBasePositionAndOrientation(sid, world_pos, world_quat)

# ========================
# 动态障碍物
# ========================
class DynObstacle:
    def __init__(self, center0, radius, color=(1,0.3,0.3,0.9),
                 motion="sinxy", amp=(0.08, 0.06, 0.0), freq=(0.25, 0.33, 0.0), phase=(0.0, 1.0, 0.0)):
        self.center0 = np.array(center0, dtype=float)
        self.radius = float(radius)
        self.amp = np.array(amp, dtype=float)
        self.freq = np.array(freq, dtype=float)
        self.phase = np.array(phase, dtype=float)
        self.motion = motion
        self.body = create_sphere(radius, mass=STATIC_MASS, color=color)
        p.setCollisionFilterGroupMask(self.body, -1, 0, 0)  # 不参与碰撞
        p.changeDynamics(self.body, -1, linearDamping=0, angularDamping=0)

    def pos_at(self, t):
        if self.motion == "sinxy":
            dx = self.amp[0]*np.sin(2*np.pi*self.freq[0]*t + self.phase[0])
            dy = self.amp[1]*np.cos(2*np.pi*self.freq[1]*t + self.phase[1])
            dz = self.amp[2]*np.sin(2*np.pi*self.freq[2]*t + self.phase[2])
            return self.center0 + np.array([dx, dy, dz])
        elif self.motion == "line":
            return self.center0 + self.amp * (t % 1.0 - 0.5) * 2.0
        else:
            return self.center0

    def update(self, t):
        pos = self.pos_at(t).tolist()
        p.resetBasePositionAndOrientation(self.body, pos, [0,0,0,1])

class ObstacleManager:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def update(self, t):
        for ob in self.obstacles:
            ob.update(t)

    def get_state(self, t):
        centers = []
        radii = []
        for ob in self.obstacles:
            centers.append(ob.pos_at(t))
            radii.append(ob.radius)
        return np.vstack(centers), np.asarray(radii, dtype=float)

# ========================
# 控制器（任务空间末端误差 + 终点避障）
# ========================
class PredictiveSamplingController7D_TS:
    def __init__(self, robot_id, tool_link, arm_joints,
                 rs_model: RobotSpheresModel,
                 N_eval, N_via, vel_lim, acc_lim, q_limits, xg,
                 dt_control, N_candidates, R_sampling,
                 lambda_avoid=LAMBDA_AVOID, safety_margin=SAFETY_MARGIN):
        self.robot = robot_id
        self.tool_link = tool_link
        self.arm_joints = arm_joints
        self.rs_model = rs_model

        self.ndof = 7
        self.vptraj = VPTraj(self.ndof, N_eval, N_via, vel_lim=np.asarray(vel_lim), acc_lim=np.asarray(acc_lim))
        self.vptraj_idle = VPTraj(self.ndof, N_eval, 1, vel_lim=np.asarray(vel_lim), acc_lim=np.asarray(acc_lim))

        self.q_limits = np.asarray(q_limits, float)  # (7,2)
        self.xg = np.asarray(xg, float)              # (3,)
        self.dt_control = float(dt_control)
        self.N_candidates = int(N_candidates)
        self.R = float(R_sampling)

        # 避障项
        self.lambda_avoid = float(lambda_avoid)
        self.safety_margin = float(safety_margin)
        self.obs_centers = None   # (K,3)
        self.obs_radii   = None   # (K,)

        self.p_next = None
        self.T_next = None

        # 记录
        self.samples_loss_log = []
        self.sol_log = []

    def set_obstacles_state(self, centers, radii):
        self.obs_centers = np.asarray(centers, dtype=float)
        self.obs_radii   = np.asarray(radii, dtype=float)

    # ------- 损失函数 -------
    def loss_fn(self, q, dq, ddq, T):
        # q: (B, N, 7), T: (B,) or scalar
        T = np.asarray(T)
        duration_cost = (np.full(q.shape[0], T) if T.ndim == 0 else T).astype(float)

        # 关节越界软惩罚
        low = self.q_limits[:, 0][None, None, :]
        up  = self.q_limits[:, 1][None, None, :]
        viol = (q < low) | (q > up)
        limit_violation_cost = 1e6 * np.sum(viol, axis=(1, 2))

        # 末端误差（终点）
        qT = q[:, -1, :]                       # (B,7)
        xT = fk_eef_pos_batch(self.robot, self.tool_link, self.arm_joints, qT)  # (B,3)
        terminal_cost = 1e3 * np.sum((xT - self.xg[None, :])**2, axis=1)

        # 终点避障（若提供障碍物）
        avoid_cost = 0.0
        if self.obs_centers is not None and self.obs_centers.size > 0:
            # 逐样本：拿 qT 下机器人所有球，与障碍物批量计算距离
            list_C, list_R = self.rs_model.batch_world_spheres_for_qT(qT)
            K = self.obs_centers.shape[0]
            margin = self.safety_margin
            batch_cost = []
            for Ci, Ri in zip(list_C, list_R):           # Ci:(M,3), Ri:(M,)
                # pairwise dist robot(M) x obs(K)
                # dist_ij = ||Ci - Oj||2
                diff = Ci[:, None, :] - self.obs_centers[None, :, :]        # (M,K,3)
                dist = np.linalg.norm(diff, axis=2)                          # (M,K)
                sum_r = Ri[:, None] + self.obs_radii[None, :] + margin       # (M,K)
                gap = dist - sum_r                                           # (M,K)
                pen = np.maximum(0.0, -gap)                                  # (M,K)
                batch_cost.append(np.sum(pen**2))
            avoid_cost = self.lambda_avoid * np.asarray(batch_cost, dtype=float)

        return terminal_cost + duration_cost + limit_violation_cost + avoid_cost

    # ------- 采样 -------
    def predictive_sampling(self, q, dq):
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

    # ------- idle -------
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
    arm_joints = env.get_movable_joints(env.robot, gripper=False)
    q_limits = np.array([get_joint_limits(env.robot, j) for j in arm_joints], dtype=float)
    dq_limits = np.array([get_max_velocity(env.robot, j) for j in arm_joints], dtype=float)
    true_amax = np.full(7, true_amax_default, dtype=float)
    _, ddq_limits = get_dynamical_limits(env.robot, arm_joints, max_accelerations=true_amax)

    # --- 工具端 link 索引 ---
    tool_link = env.get_tool_link(env.robot)

    # --- 初值（与仿真对齐）---
    q_full = np.array(env.get_joint_positions(env.robot), dtype=float)  # 7臂+2指
    q_arm = q_full[:7].copy()
    dq_arm = np.zeros_like(q_arm)

    # --- 任务空间目标（示例：直接指定）---
    xg = np.array([0.55, 0.00, 0.70], dtype=float)
    env.render_pose(xg.tolist())

    # --- 读取 YAML 的碰撞小球模型 ---
    with open(f"{COLLISION_MODELS_PATH}/franka.yml", 'r') as f:
        collision_spheres = yaml.safe_load(f)

    # --- 创建“跟随显示”的小球（仅可视化） ---
    vis = CollisionSpheresVis(env.robot, color=(0.2, 0.6, 1.0, 0.6))
    vis.add_from_spec(collision_spheres)

    # --- 机器人小球模型（用于在 loss 中计算距离） ---
    rs_model = RobotSpheresModel(env.robot, arm_joints, collision_spheres)

    # --- 创建 2-3 个动态障碍物 ---
    # 注意高度 Z 与 Panda 工作空间匹配；可按需要调整
    obs_list = [
        DynObstacle(center0=[0.45,  0.10, 0.75], radius=0.06, color=(1,0.2,0.2,0.9),
                    motion="sinxy", amp=(0.06, 0.05, 0.0), freq=(0.25, 0.32, 0.0), phase=(0.0, 0.5, 0.0)),
        DynObstacle(center0=[0.70, -0.20, 0.85], radius=0.07, color=(1,0.6,0.2,0.9),
                    motion="sinxy", amp=(0.05, 0.04, 0.0), freq=(0.20, 0.28, 0.0), phase=(0.3, 1.0, 0.0)),
        DynObstacle(center0=[0.55, -0.05, 0.70], radius=0.05, color=(0.9,0.3,1.0,0.9),
                    motion="sinxy", amp=(0.04, 0.06, 0.0), freq=(0.30, 0.18, 0.0), phase=(1.0, 0.2, 0.0)),
    ]
    obs_mgr = ObstacleManager(obs_list)
    # 先更新一次（t=0）以显示在正确位置
    obs_mgr.update(0.0)

    # --- 控制器（加入避障） ---
    controller = PredictiveSamplingController7D_TS(
        robot_id=env.robot, tool_link=tool_link, arm_joints=arm_joints,
        rs_model=rs_model,
        N_eval=N_eval, N_via=N_via,
        vel_lim=dq_limits, acc_lim=ddq_limits,
        q_limits=q_limits, xg=xg,
        dt_control=dt_control, N_candidates=N_candidates, R_sampling=R_sampling,
        lambda_avoid=LAMBDA_AVOID, safety_margin=SAFETY_MARGIN
    )

    # --- 末端轨迹可视化 ---
    eef_prev = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
    eef_hist = [eef_prev.copy()]

    # --- 主循环 ---
    total_time, acc = 0.0, 0.0
    steps = 0
    print("[INFO] Running task-space predictive sampling (no IK) with dynamic obstacles...")
    while p.isConnected() and total_time < sim_duration:
        acc += env.dt
        total_time += env.dt

        # 障碍物更新（每个仿真步）
        obs_mgr.update(total_time)

        if acc + 1e-9 >= dt_control:
            acc -= dt_control

            # 把“当前障碍物状态”送进控制器（终点避障按当前帧估计）
            obs_centers, obs_radii = obs_mgr.get_state(total_time)
            controller.set_obstacles_state(obs_centers, obs_radii)

            # 控制一拍
            q_arm, dq_arm = controller.control(q_arm, dq_arm)

            # 写回仿真（只改 7 轴）
            q_full[:7] = q_arm
            q_full[-2:] = gripper_opening
            env.set_joint_positions(env.robot, q_full.tolist()[:7], gripper=False)

            # 末端轨迹线
            eef_now = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
            eef_hist.append(eef_now.copy())
            p.addUserDebugLine(eef_prev, eef_now, lineWidth=2, lineColorRGB=GREEN[:3], lifeTime=1.5)
            eef_prev = eef_now.copy()

            # 最近“机器人球-障碍球”可视诊断
            if DRAW_CLOSEST_PAIR:
                # 取当前 q_arm 下的机器人小球
                rob_C, rob_R = rs_model.world_spheres_for_q(q_arm)
                # pairwise
                diff = rob_C[:, None, :] - obs_centers[None, :, :]
                dist = np.linalg.norm(diff, axis=2)                    # (M,K)
                sum_r = rob_R[:, None] + obs_radii[None, :] + SAFETY_MARGIN
                gap = dist - sum_r
                i_m, j_m = np.unravel_index(np.argmin(gap), gap.shape)
                c1 = rob_C[i_m]; c2 = obs_centers[j_m]
                p.addUserDebugLine(c1, c2, lineWidth=3, lineColorRGB=[1,0,0], lifeTime=CLOSEST_LINE_LIFE)
                if steps % int(max(1, 1.0/dt_control)) == 0:
                    print(f"[t={total_time:5.2f}s] cartesian dist={np.linalg.norm(eef_now-xg):.3f} m | min_gap={gap[i_m,j_m]:+.3f} m", end="\r")

            steps += 1

        # 跟随显示的关节小球
        vis.update()
        env.step(sleep=True)

    print("\n[INFO] Done.")
    # 结束统计
    eef_final = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
    print(f"[RESULT] Final EEF dist to goal = {np.linalg.norm(eef_final - xg):.6f} m")

    # 收敛曲线
    eef_hist = np.array(eef_hist, dtype=float)
    plt.figure()
    plt.plot(np.linalg.norm(eef_hist - xg[None, :], axis=1))
    plt.xlabel("Step"); plt.ylabel("EEF to Goal Dist (m)")
    plt.title("End-Effector Distance to Goal Over Time (with dynamic obstacles)")
    plt.grid(); plt.show()
