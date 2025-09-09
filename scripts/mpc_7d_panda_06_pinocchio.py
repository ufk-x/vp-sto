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

# === Pinocchio ===
import pinocchio as pin

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
LAMBDA_AVOID   = 1.0e4
SAFETY_MARGIN  = 0.05

# 可视化
DRAW_CLOSEST_PAIR = True
CLOSEST_LINE_LIFE = 0.15

# 7 轴关节名（按你控制的 q 顺序）
ARM_JOINT_NAMES = [
    "panda_joint1","panda_joint2","panda_joint3",
    "panda_joint4","panda_joint5","panda_joint6","panda_joint7"
]
# 末端 link 名（常见：panda_grasptarget；若你的 URDF 没有，会自动回退到 panda_link8 / panda_hand）
EEF_FRAME_CANDIDATES = ["panda_grasptarget", "panda_link8", "panda_hand"]

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
# Pinocchio 运动学封装
# ========================
class PinKinematics:
    def __init__(self, urdf_path, arm_joint_names, eef_frame_candidates):
        # 固定基加载
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # 关节名 -> (jointId, q-index)
        self.joint_id = {}
        self.q_index  = {}
        for name in arm_joint_names:
            jid = self.model.getJointId(name)
            assert jid != 0, f"Joint name not found in URDF: {name}"
            self.joint_id[name] = jid
            self.q_index[name]  = self.model.idx_qs[jid]  # revolute -> 单自由度

        # 末端 frame
        self.eef_frame_id = None
        for fname in eef_frame_candidates:
            try:
                self.eef_frame_id = self.model.getFrameId(fname)
                break
            except Exception:
                continue
        if self.eef_frame_id is None:
            raise ValueError(f"None of EEF frames found: {eef_frame_candidates}")

        self.nq = self.model.nq
        # 中性位姿（finger 等其余自由度走 neutral）
        self.q_neutral = pin.neutral(self.model).copy()

    def build_q_full(self, q_arm7):
        """把 7D 关节写进完整 nq 向量，其余保持 neutral。"""
        q = self.q_neutral.copy()
        for name, v in zip(ARM_JOINT_NAMES, q_arm7):
            q[self.q_index[name]] = float(v)
        return q

    def eef_pos_batch(self, q_batch):
        """q_batch: (B,7) -> (B,3)"""
        out = []
        for q7 in q_batch:
            q = self.build_q_full(q7)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            T = self.data.oMf[self.eef_frame_id]
            out.append(T.translation.copy())
        return np.asarray(out, dtype=float)

    def frames_SE3_for_q(self, q7, frame_ids):
        """给定 q7，返回所需 frame 的世界 SE3 列表（按 frame_ids 顺序）。"""
        q = self.build_q_full(q7)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return [self.data.oMf[fid] for fid in frame_ids]

# ========================
# 机器人小球模型（Pinocchio 版）
# ========================
class RobotSpheresModel:
    def __init__(self, pin_kin: PinKinematics, spec_dict):
        """
        spec_dict: {'link_name': [{'center':[x,y,z], 'radius':r}, ...], ...}
        """
        self.pin = pin_kin
        self.frame_ids = []
        self.local_centers_list = []
        self.local_radii_list   = []

        for link_name, spheres in spec_dict.items():
            try:
                fid = self.pin.model.getFrameId(link_name)
            except Exception:
                raise ValueError(f"[Pinocchio] Frame not found in URDF: {link_name}")
            self.frame_ids.append(fid)
            centers = np.array([s["center"] for s in spheres], dtype=float)  # (Ni,3)
            radii   = np.array([s["radius"] for s in spheres], dtype=float)  # (Ni,)
            self.local_centers_list.append(centers)
            self.local_radii_list.append(radii)

        self.num_links = len(self.frame_ids)
        self.num_spheres = int(sum(len(r) for r in self.local_radii_list))

    def world_spheres_for_q(self, q7):
        """
        返回：
          centers_world: (M,3)
          radii:         (M,)
        """
        Ts = self.pin.frames_SE3_for_q(q7, self.frame_ids)
        centers_w, radii_w = [], []
        for T, C_local, R_local in zip(Ts, self.local_centers_list, self.local_radii_list):
            # T.act(p_local) -> p_world
            for c, r in zip(C_local, R_local):
                centers_w.append(T.act(c))
                radii_w.append(r)
        return np.asarray(centers_w, dtype=float), np.asarray(radii_w, dtype=float)

    def batch_world_spheres_for_qT(self, qT_batch):
        list_centers, list_radii = [], []
        for q in qT_batch:
            C, R = self.world_spheres_for_q(q)
            list_centers.append(C); list_radii.append(R)
        return list_centers, list_radii

# ========================
# 可视化：跟随 link 的小球（仅显示用途；读 PyBullet 可见机器人）
# ========================
class CollisionSpheresVis:
    def __init__(self, robot, color=(0.2, 0.6, 1.0, 0.6)):
        self.robot = robot
        self.color = color
        self._items = []  # {link, local_centers(N,3), radii(N,), sphere_ids(N,)}

    def add_from_spec(self, spec: dict):
        for link_name, spheres in spec.items():
            link = link_from_name(self.robot, link_name)
            local_centers, radii, ids = [], [], []
            for s in spheres:
                c = np.asarray(s["center"], dtype=float)
                r = float(s["radius"])
                sid = create_sphere(r, mass=STATIC_MASS, color=self.color)
                # 仅显示，不碰撞/不摩擦
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
        p.setCollisionFilterGroupMask(self.body, -1, 0, 0)
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
        p.resetBasePositionAndOrientation(self.body, self.pos_at(t).tolist(), [0,0,0,1])

class ObstacleManager:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def update(self, t):
        for ob in self.obstacles:
            ob.update(t)

    def get_state(self, t):
        centers = [ob.pos_at(t) for ob in self.obstacles]
        radii   = [ob.radius for ob in self.obstacles]
        return np.vstack(centers), np.asarray(radii, dtype=float)

# ========================
# 控制器（任务空间末端误差 + 终点避障，FK 由 Pinocchio 计算）
# ========================
class PredictiveSamplingController7D_TS:
    def __init__(self, robot_id, arm_joints,
                 pin_kin: PinKinematics, rs_model: RobotSpheresModel,
                 N_eval, N_via, vel_lim, acc_lim, q_limits, xg,
                 dt_control, N_candidates, R_sampling,
                 lambda_avoid=LAMBDA_AVOID, safety_margin=SAFETY_MARGIN):
        self.robot = robot_id                 # 仅用于写回控制（渲染）
        self.arm_joints = arm_joints

        self.pin = pin_kin
        self.rs_model = rs_model

        self.ndof = 7
        self.vptraj = VPTraj(self.ndof, N_eval, N_via, vel_lim=np.asarray(vel_lim), acc_lim=np.asarray(acc_lim))
        self.vptraj_idle = VPTraj(self.ndof, N_eval, 1, vel_lim=np.asarray(vel_lim), acc_lim=np.asarray(acc_lim))

        self.q_limits = np.asarray(q_limits, float)
        self.xg = np.asarray(xg, float)
        self.dt_control = float(dt_control)
        self.N_candidates = int(N_candidates)
        self.R = float(R_sampling)

        self.lambda_avoid = float(lambda_avoid)
        self.safety_margin = float(safety_margin)
        self.obs_centers = None
        self.obs_radii   = None

        self.p_next = None
        self.T_next = None

        self.samples_loss_log = []
        self.sol_log = []

    def set_obstacles_state(self, centers, radii):
        self.obs_centers = np.asarray(centers, dtype=float)
        self.obs_radii   = np.asarray(radii, dtype=float)

    # ------- 损失函数 -------
    def loss_fn(self, q, dq, ddq, T):
        # 时间代价
        T = np.asarray(T)
        duration_cost = (np.full(q.shape[0], T) if T.ndim == 0 else T).astype(float)

        # 关节越界软惩罚
        low = self.q_limits[:, 0][None, None, :]
        up  = self.q_limits[:, 1][None, None, :]
        viol = (q < low) | (q > up)
        limit_violation_cost = 1e6 * np.sum(viol, axis=(1, 2))

        # 末端误差（终点）—— 用 Pinocchio
        qT = q[:, -1, :]                       # (B,7)
        xT = self.pin.eef_pos_batch(qT)        # (B,3)
        terminal_cost = 1e3 * np.sum((xT - self.xg[None, :])**2, axis=1)

        # 终点避障
        avoid_cost = 0.0
        if self.obs_centers is not None and self.obs_centers.size > 0:
            list_C, list_R = self.rs_model.batch_world_spheres_for_qT(qT)
            margin = self.safety_margin
            batch_cost = []
            for Ci, Ri in zip(list_C, list_R):           # Ci:(M,3), Ri:(M,)
                diff = Ci[:, None, :] - self.obs_centers[None, :, :]        # (M,K,3)
                dist = np.linalg.norm(diff, axis=2)                          # (M,K)
                sum_r = Ri[:, None] + self.obs_radii[None, :] + margin       # (M,K)
                gap = dist - sum_r
                pen = np.maximum(0.0, -gap)
                batch_cost.append(np.sum(pen**2))
            avoid_cost = self.lambda_avoid * np.asarray(batch_cost, dtype=float)

        return terminal_cost + duration_cost + limit_violation_cost + avoid_cost

    # ------- 采样 / 复用 / idle / 主控 -------
    def predictive_sampling(self, q, dq):
        pos, vel, acc, p, T = self.vptraj.sample_trajectories(
            self.N_candidates, q, dq0=dq, qT=None, dqT=np.zeros_like(dq), Q=None, R=self.R
        )
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], T)
        self.samples_loss_log.append(loss)
        i_best = int(np.argmin(loss))
        return p[i_best], float(loss[i_best]), float(T[i_best])

    def previous_sol(self, q, dq):
        if self.p_next is None:
            return None, np.inf, 0.0
        pos, vel, acc = self.vptraj.get_trajectory(self.p_next, q, dq0=dq, qT=None, dqT=np.zeros_like(dq), T=self.T_next)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [self.T_next])[0]
        return self.p_next, float(loss), float(self.T_next)

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

    # --- 初值（与仿真对齐）---
    q_full = np.array(env.get_joint_positions(env.robot), dtype=float)  # 7臂+2指
    q_arm = q_full[:7].copy()
    dq_arm = np.zeros_like(q_arm)

    # --- 任务空间目标（示例：直接指定）---
    xg = np.array([0.55, 0.00, 0.75], dtype=float)
    env.render_pose(xg.tolist())

    # --- 读取 YAML 的碰撞小球模型 ---
    with open(f"{COLLISION_MODELS_PATH}/franka.yml", 'r') as f:
        collision_spheres = yaml.safe_load(f)

    # --- 创建“跟随显示”的小球（仅可视化） ---
    vis = CollisionSpheresVis(env.robot, color=(0.2, 0.6, 1.0, 0.6))
    vis.add_from_spec(collision_spheres)

    # --- Pinocchio 运动学 ---
    pin_kin = PinKinematics(PANDA_URDF, ARM_JOINT_NAMES, EEF_FRAME_CANDIDATES)

    # --- 机器人小球模型（用于在 loss 中计算距离，Pinocchio 版） ---
    rs_model = RobotSpheresModel(pin_kin, collision_spheres)

    # --- 创建 2-3 个动态障碍物 ---
    obs_list = [
        DynObstacle(center0=[0.45,  0.10, 0.75], radius=0.06, color=(1,0.2,0.2,0.9),
                    motion="sinxy", amp=(0.06, 0.05, 0.0), freq=(0.25, 0.32, 0.0), phase=(0.0, 0.5, 0.0)),
    ]
    obs_mgr = ObstacleManager(obs_list)
    obs_mgr.update(0.0)

    # --- 控制器（加入避障 + Pinocchio FK） ---
    controller = PredictiveSamplingController7D_TS(
        robot_id=env.robot, arm_joints=arm_joints,
        pin_kin=pin_kin, rs_model=rs_model,
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
    print("[INFO] Running task-space predictive sampling (no IK) with dynamic obstacles, Pinocchio FK...")
    while p.isConnected() and total_time < sim_duration:
        acc += env.dt
        total_time += env.dt

        # 障碍物更新（每个仿真步）
        obs_mgr.update(total_time)

        if acc + 1e-9 >= dt_control:
            acc -= dt_control

            # 障碍状态送入控制器
            obs_centers, obs_radii = obs_mgr.get_state(total_time)
            controller.set_obstacles_state(obs_centers, obs_radii)

            # 控制一拍
            q_arm, dq_arm = controller.control(q_arm, dq_arm)

            # 写回仿真（只改 7 轴）
            q_full[:7] = q_arm
            q_full[-2:] = gripper_opening
            env.set_joint_positions(env.robot, q_full.tolist()[:7], gripper=False)

            # 末端轨迹线（从仿真读取，用于可视化）
            eef_now = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
            eef_hist.append(eef_now.copy())
            p.addUserDebugLine(eef_prev, eef_now, lineWidth=2, lineColorRGB=GREEN[:3], lifeTime=1.5)
            eef_prev = eef_now.copy()

            # 最近“机器人球-障碍球”的可视诊断（使用当前 q_arm ——用Pinocchio）
            if DRAW_CLOSEST_PAIR:
                rob_C, rob_R = rs_model.world_spheres_for_q(q_arm)  # Pinocchio 计算
                diff = rob_C[:, None, :] - obs_centers[None, :, :]
                dist = np.linalg.norm(diff, axis=2)                    # (M,K)
                sum_r = rob_R[:, None] + obs_radii[None, :] + SAFETY_MARGIN
                gap = dist - sum_r
                i_m, j_m = np.unravel_index(np.argmin(gap), gap.shape)
                p.addUserDebugLine(rob_C[i_m], obs_centers[j_m], lineWidth=3, lineColorRGB=[1,0,0], lifeTime=CLOSEST_LINE_LIFE)

                if steps % int(max(1, 1.0/dt_control)) == 0:
                    print(f"[t={total_time:5.2f}s] cartesian dist={np.linalg.norm(eef_now-xg):.3f} m | min_gap={gap[i_m,j_m]:+.3f} m", end="\r")

            steps += 1

        # 跟随显示的小球（读 PyBullet 当前姿态）
        vis.update()
        env.step(sleep=True)

    print("\n[INFO] Done.")
    eef_final = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
    print(f"[RESULT] Final EEF dist to goal = {np.linalg.norm(eef_final - xg):.6f} m")

    # 收敛曲线
    eef_hist = np.array(eef_hist, dtype=float)
    plt.figure()
    plt.plot(np.linalg.norm(eef_hist - xg[None, :], axis=1))
    plt.xlabel("Step"); plt.ylabel("EEF to Goal Dist (m)")
    plt.title("End-Effector Distance to Goal Over Time (with dynamic obstacles, Pinocchio FK)")
    plt.grid(); plt.show()
