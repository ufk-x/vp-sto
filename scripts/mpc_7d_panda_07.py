"""
与上一版区别：
- ✅ 全时刻避障：对每条候选轨迹的每一离散时刻，按该时刻的障碍物位置做距离铰链惩罚并累加
- 仍使用 Pinocchio 做 FK（末端 & 小球），不修改仿真机器人状态
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
    get_link_pose, link_from_name,
    set_joint_positions as set_joint_positions_util,
    create_sphere, STATIC_MASS,
)
from planners.vptraj import VPTraj
from matplotlib import pyplot as plt
import pinocchio as pin

# ========================
# 参数
# ========================
N_via = 4
N_candidates = 100
N_eval = 50
dt_control = 0.05
sim_duration = 8.0

R_sampling = 1e1
true_amax_default = 3.0
gripper_opening = 0.06

# 避障代价参数
LAMBDA_AVOID   = 1.0e4
SAFETY_MARGIN  = 0.05
AVOID_STRIDE   = 5        # 每隔多少个离散时刻评估一次（1=全时刻；2=隔一个；3=隔两个...）

# 可视化
DRAW_CLOSEST_PAIR = True
CLOSEST_LINE_LIFE = 0.15

ARM_JOINT_NAMES = [
    "panda_joint1","panda_joint2","panda_joint3",
    "panda_joint4","panda_joint5","panda_joint6","panda_joint7"
]
EEF_FRAME_CANDIDATES = ["panda_grasptarget", "panda_link8", "panda_hand"]

# ========================
# 工具
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
# Pinocchio 运动学
# ========================
class PinKinematics:
    def __init__(self, urdf_path, arm_joint_names, eef_frame_candidates):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.joint_id = {}
        self.q_index  = {}
        for name in arm_joint_names:
            jid = self.model.getJointId(name)
            assert jid != 0, f"Joint name not found: {name}"
            self.joint_id[name] = jid
            self.q_index[name]  = self.model.idx_qs[jid]
        self.eef_frame_id = None
        for fname in eef_frame_candidates:
            try:
                self.eef_frame_id = self.model.getFrameId(fname); break
            except Exception:
                continue
        if self.eef_frame_id is None:
            raise ValueError(f"EEF frame not found in {eef_frame_candidates}")
        self.nq = self.model.nq
        self.q_neutral = pin.neutral(self.model).copy()

    def build_q_full(self, q7):
        q = self.q_neutral.copy()
        for nm, v in zip(ARM_JOINT_NAMES, q7):
            q[self.q_index[nm]] = float(v)
        return q

    def eef_pos_batch(self, q_batch):
        out = []
        for q7 in q_batch:
            q = self.build_q_full(q7)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            T = self.data.oMf[self.eef_frame_id]
            out.append(T.translation.copy())
        return np.asarray(out, dtype=float)

    def frames_SE3_for_q(self, q7, frame_ids):
        q = self.build_q_full(q7)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return [self.data.oMf[fid] for fid in frame_ids]

# ========================
# 小球模型（Pinocchio）
# ========================
class RobotSpheresModel:
    def __init__(self, pin_kin: PinKinematics, spec_dict):
        self.pin = pin_kin
        self.frame_ids = []
        self.local_centers_list = []
        self.local_radii_list   = []
        for link_name, spheres in spec_dict.items():
            fid = self.pin.model.getFrameId(link_name)
            self.frame_ids.append(fid)
            self.local_centers_list.append(np.array([s["center"] for s in spheres], dtype=float))
            self.local_radii_list.append(np.array([s["radius"] for s in spheres], dtype=float))

    def world_spheres_for_q(self, q7):
        Ts = self.pin.frames_SE3_for_q(q7, self.frame_ids)
        centers_w, radii_w = [], []
        for T, C_local, R_local in zip(Ts, self.local_centers_list, self.local_radii_list):
            for c, r in zip(C_local, R_local):
                centers_w.append(T.act(c))
                radii_w.append(r)
        return np.asarray(centers_w, dtype=float), np.asarray(radii_w, dtype=float)

# ========================
# 可视化：link-跟随显示球
# ========================
class CollisionSpheresVis:
    def __init__(self, robot, color=(0.2, 0.6, 1.0, 0.6)):
        self.robot = robot
        self.color = color
        self._items = []

    def add_from_spec(self, spec: dict):
        for link_name, spheres in spec.items():
            link = link_from_name(self.robot, link_name)
            local_centers, radii, ids = [], [], []
            for s in spheres:
                c = np.asarray(s["center"], dtype=float)
                r = float(s["radius"])
                sid = create_sphere(r, mass=STATIC_MASS, color=self.color)
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
                world_pos, world_quat = p.multiplyTransforms(link_pos, link_quat, c_local.tolist(), [0,0,0,1])
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

    def get_state_over_times(self, ts_abs):
        """ts_abs: (N,) 绝对时间数组 -> 返回 (N,K,3) centers_seq, (K,) radii"""
        K = len(self.obstacles)
        centers_seq = np.zeros((len(ts_abs), K, 3), dtype=float)
        radii = np.array([ob.radius for ob in self.obstacles], dtype=float)
        for i, t in enumerate(ts_abs):
            for k, ob in enumerate(self.obstacles):
                centers_seq[i, k, :] = ob.pos_at(t)
        return centers_seq, radii

# ========================
# 控制器：全时刻避障
# ========================
class PredictiveSamplingController7D_TS:
    def __init__(self, robot_id, arm_joints,
                 pin_kin: PinKinematics, rs_model: RobotSpheresModel,
                 N_eval, N_via, vel_lim, acc_lim, q_limits, xg,
                 dt_control, N_candidates, R_sampling,
                 lambda_avoid=LAMBDA_AVOID, safety_margin=SAFETY_MARGIN,
                 avoid_stride=AVOID_STRIDE):
        self.robot = robot_id
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
        self.avoid_stride = int(avoid_stride)

        # 动态障碍轨迹提供器：provider(ts_abs) -> (N,K,3),(K,)
        self.obs_traj_provider = None
        self.t_now = 0.0

        self.p_next = None
        self.T_next = None
        self.samples_loss_log = []
        self.sol_log = []

    def set_obstacles_trajectory_provider(self, provider_callable, t_now):
        """
        provider_callable(ts_abs: np.ndarray) -> (centers_seq:(N,K,3), radii:(K,))
        t_now: 当前绝对时间（秒）
        """
        self.obs_traj_provider = provider_callable
        self.t_now = float(t_now)

    # ------- 损失函数（含全时刻避障）-------
    def loss_fn(self, q, dq, ddq, T):
        """
        q:  (B, N, 7)  —— 注意：这里传入的是 pos[:, 1:], 即不含初始点
        T:  (B,) 或标量，对每个候选的轨迹时长
        """
        B, N, _ = q.shape
        T = np.asarray(T)
        duration_cost = (np.full(B, T) if T.ndim == 0 else T).astype(float)

        # 关节越界软惩罚
        low = self.q_limits[:, 0][None, None, :]
        up  = self.q_limits[:, 1][None, None, :]
        viol = (q < low) | (q > up)
        limit_violation_cost = 1e6 * np.sum(viol, axis=(1, 2))

        # 末端终点误差（Pinocchio）
        qT = q[:, -1, :]
        xT = self.pin.eef_pos_batch(qT)
        terminal_cost = 1e3 * np.sum((xT - self.xg[None, :])**2, axis=1)

        # 全时刻避障：对每个候选、每个离散时刻（可 stride）累加
        avoid_cost = np.zeros((B,), dtype=float)
        if self.obs_traj_provider is not None:
            stride = max(1, self.avoid_stride)
            for b in range(B):
                # 该候选的每个离散时刻在 [0, T[b]] 的相对时间
                ts_rel = np.linspace(0.0, float(duration_cost[b]), N)
                ts_rel = ts_rel[::stride]
                ts_abs = self.t_now + ts_rel             # 绝对时间
                centers_seq, radii = self.obs_traj_provider(ts_abs)  # (n_eval,K,3),(K,)

                # 逐时刻：Pinocchio 计算机器人球 -> pairwise hinge 距离
                acc_cost = 0.0
                for ii, n in enumerate(range(0, N, stride)):
                    qn = q[b, n, :]       # (7,)
                    rob_C, rob_R = self.rs_model.world_spheres_for_q(qn)
                    diff = rob_C[:, None, :] - centers_seq[ii][None, :, :]   # (M,K,3)
                    dist = np.linalg.norm(diff, axis=2)                       # (M,K)
                    sum_r = rob_R[:, None] + radii[None, :] + self.safety_margin
                    gap = dist - sum_r
                    pen = np.maximum(0.0, -gap)
                    acc_cost += np.sum(pen**2)
                avoid_cost[b] = self.lambda_avoid * acc_cost

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
    # 环境
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=True, mp4=None)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

    # Panda 7 轴与限幅
    arm_joints = env.get_movable_joints(env.robot, gripper=False)
    q_limits = np.array([get_joint_limits(env.robot, j) for j in arm_joints], dtype=float)
    dq_limits = np.array([get_max_velocity(env.robot, j) for j in arm_joints], dtype=float)
    true_amax = np.full(7, true_amax_default, dtype=float)
    _, ddq_limits = get_dynamical_limits(env.robot, arm_joints, max_accelerations=true_amax)

    # 初值
    q_full = np.array(env.get_joint_positions(env.robot), dtype=float)
    q_arm = q_full[:7].copy()
    dq_arm = np.zeros_like(q_arm)

    # 目标
    xg = np.array([0.55, 0.00, 0.70], dtype=float)
    env.render_pose(xg.tolist())

    # 碰撞小球（YAML）
    with open(f"{COLLISION_MODELS_PATH}/franka.yml", 'r') as f:
        collision_spheres = yaml.safe_load(f)

    # 可视化跟随球
    vis = CollisionSpheresVis(env.robot, color=(0.2, 0.6, 1.0, 0.6))
    vis.add_from_spec(collision_spheres)

    # Pinocchio 运动学与小球模型
    pin_kin = PinKinematics(PANDA_URDF, ARM_JOINT_NAMES, EEF_FRAME_CANDIDATES)
    rs_model = RobotSpheresModel(pin_kin, collision_spheres)

    # 动态障碍
    obs_list = [
        DynObstacle(center0=[0.45,  0.10, 0.75], radius=0.06, color=(1,0.2,0.2,0.9),
                    motion="sinxy", amp=(0.06, 0.05, 0.0), freq=(0.25, 0.32, 0.0), phase=(0.0, 0.5, 0.0)),
        DynObstacle(center0=[0.70, -0.20, 0.85], radius=0.07, color=(1,0.6,0.2,0.9),
                    motion="sinxy", amp=(0.05, 0.04, 0.0), freq=(0.20, 0.28, 0.0), phase=(0.3, 1.0, 0.0)),
        DynObstacle(center0=[0.55, -0.05, 0.70], radius=0.05, color=(0.9,0.3,1.0,0.9),
                    motion="sinxy", amp=(0.04, 0.06, 0.0), freq=(0.30, 0.18, 0.0), phase=(1.0, 0.2, 0.0)),
    ]
    obs_mgr = ObstacleManager(obs_list)
    obs_mgr.update(0.0)

    # 控制器（全时刻避障）
    controller = PredictiveSamplingController7D_TS(
        robot_id=env.robot, arm_joints=arm_joints,
        pin_kin=pin_kin, rs_model=rs_model,
        N_eval=N_eval, N_via=N_via,
        vel_lim=dq_limits, acc_lim=ddq_limits,
        q_limits=q_limits, xg=xg,
        dt_control=dt_control, N_candidates=N_candidates, R_sampling=R_sampling,
        lambda_avoid=LAMBDA_AVOID, safety_margin=SAFETY_MARGIN,
        avoid_stride=AVOID_STRIDE
    )

    # 提供“障碍物轨迹查询”给控制器（按绝对时间批量查询）
    def obs_traj_provider(ts_abs: np.ndarray):
        centers_seq, radii = obs_mgr.get_state_over_times(ts_abs)
        return centers_seq, radii
    controller.set_obstacles_trajectory_provider(obs_traj_provider, t_now=0.0)

    # 末端轨迹可视化
    eef_prev = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
    eef_hist = [eef_prev.copy()]

    total_time, acc = 0.0, 0.0
    steps = 0
    print("[INFO] Running task-space predictive sampling (full-time avoidance, Pinocchio FK)...")
    while p.isConnected() and total_time < sim_duration:
        acc += env.dt
        total_time += env.dt

        # 动态障碍更新（渲染）
        obs_mgr.update(total_time)

        if acc + 1e-9 >= dt_control:
            acc -= dt_control

            # 告诉控制器当前绝对时间（用于预测区间的 t_now）
            controller.set_obstacles_trajectory_provider(obs_traj_provider, t_now=total_time)

            # 控制一拍
            q_arm, dq_arm = controller.control(q_arm, dq_arm)

            # 写回仿真（只 7 轴）
            q_full[:7] = q_arm
            q_full[-2:] = gripper_opening
            env.set_joint_positions(env.robot, q_full.tolist()[:7], gripper=False)

            # 末端轨迹线
            eef_now = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
            eef_hist.append(eef_now.copy())
            p.addUserDebugLine(eef_prev, eef_now, lineWidth=2, lineColorRGB=GREEN[:3], lifeTime=1.5)
            eef_prev = eef_now.copy()

            # 诊断线：最近“机器人球-障碍球”
            if DRAW_CLOSEST_PAIR:
                # 取当前时刻障碍
                obs_centers_cur, obs_radii_cur = obs_mgr.get_state(total_time)
                rob_C, rob_R = rs_model.world_spheres_for_q(q_arm)  # Pin 计算
                diff = rob_C[:, None, :] - obs_centers_cur[None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                sum_r = rob_R[:, None] + obs_radii_cur[None, :] + SAFETY_MARGIN
                gap = dist - sum_r
                i_m, j_m = np.unravel_index(np.argmin(gap), gap.shape)
                p.addUserDebugLine(rob_C[i_m], obs_centers_cur[j_m], lineWidth=3, lineColorRGB=[1,0,0], lifeTime=CLOSEST_LINE_LIFE)

                if steps % int(max(1, 1.0/dt_control)) == 0:
                    print(f"[t={total_time:5.2f}s] ‖eef-xg‖={np.linalg.norm(eef_now-xg):.3f} m | min_gap(now)={gap[i_m,j_m]:+.3f} m", end="\r")

            steps += 1

        vis.update()
        env.step(sleep=True)

    print("\n[INFO] Done.")
    eef_final = np.array(env.get_eef_pose(env.robot)[:3], dtype=float)
    print(f"[RESULT] Final EEF dist to goal = {np.linalg.norm(eef_final - xg):.6f} m")

    eef_hist = np.array(eef_hist, dtype=float)
    plt.figure()
    plt.plot(np.linalg.norm(eef_hist - xg[None, :], axis=1))
    plt.xlabel("Step"); plt.ylabel("EEF to Goal Dist (m)")
    plt.title("End-Effector Distance to Goal Over Time (full-time avoidance)")
    plt.grid(); plt.show()
