"""
VP-STO MPC Demo with Animation Saving Support

This script demonstrates a VP-STO based MPC controller with real-time trajectory visualization.
The animation can be saved as MP4 or GIF format.

Usage:
1. Set SAVE_ANIMATION = True and ANIMATION_FILENAME to automatically save
2. Or run interactively and choose whether to save when prompted

Requirements for saving:
- For MP4: ffmpeg (install with: sudo apt-get install ffmpeg)  
- For GIF: pillow (install with: pip install pillow)
"""

import os
import sys
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

# Configuration for animation saving
SAVE_ANIMATION = False  # Set to True to automatically save animation
ANIMATION_FILENAME = "mpc_animation.mp4"  # Default filename (supports .mp4 or .gif)

# --- optional project path (keeps compatibility with your repo layout) ------
try:
    from config import PLANNER_PATH
    if PLANNER_PATH not in sys.path:
        sys.path.append(PLANNER_PATH)
except Exception:
    pass

# --- VP-STO core -------------------------------------------------------------
from planners.vpsto import VPSTO, VPSTOOptions
from planners.vptraj import VPTraj

# --- exact polygon collision if shapely is available ------------------------
try:
    from shapely.geometry import Polygon, MultiPolygon, LineString
except Exception:
    Polygon = MultiPolygon = LineString = None


# =============================================================================
# Environment: same static polygons as your original SET demo
# =============================================================================
class CollisionEnvironment:
    def __init__(self):
        self.poly_list = []
        # # 左下区域
        # self.poly_list.append(np.array([
        #     [0.15, 0.2],   # 顶点1
        #     [0.05, 0.2],   # 顶点2  
        #     [0.05, 0.1],   # 顶点3
        #     [0.15, 0.1],   # 顶点4
        #     [0.2, 0.1],    # 顶点5
        #     [0.2, 0.4],
        #     [0.05, 0.4],
        #     [0.05, 0.3],
        #     [0.15, 0.3],
        # ]))
        # # 中上区域
        # self.poly_list.append(np.array([
        #     [0.4, 0.2],   # 顶点1
        #     [0.3, 0.2],   # 顶点2  
        #     [0.3, 0.1],   # 顶点3
        #     [0.4, 0.1],   # 顶点4
        #     [0.45, 0.1],  # 顶点5
        #     [0.45, 0.4],
        #     [0.3, 0.4],
        #     [0.3, 0.3],
        #     [0.4, 0.3],
        # ]))
        self.poly_list.append(np.array([
            [0.2, 0.0],   # 顶点1
            [0.3, 0.0],   # 顶点2  
            [0.3, 0.25],   # 顶点3
            [0.2, 0.25],   # 顶点4
        ]))
        # 中上区域
        self.poly_list.append(np.array([
            [0.2, 0.27],  # 顶点1
            [0.3, 0.27],  # 顶点2
            [0.3, 0.5],  # 顶点3
            [0.2, 0.5],  # 顶点4
        ]))

        if Polygon is not None:
            self.multi_poly = MultiPolygon([Polygon(p) for p in self.poly_list])
        else:
            self.multi_poly = None
            # fallback: AABB list to detect obvious collisions when shapely is missing
            self.aabbs = []
            for p in self.poly_list:
                xmn, ymn = p.min(0); xmx, ymx = p.max(0)
                self.aabbs.append((xmn, ymn, xmx, ymx))

    def collision_length(self, pts: np.ndarray) -> float:
        if self.multi_poly is None:
            # coarse fallback: if any point falls into any polygon AABB → treat as positive length
            for xmn, ymn, xmx, ymx in self.aabbs:
                if np.any((pts[:,0] >= xmn) & (pts[:,0] <= xmx) & (pts[:,1] >= ymn) & (pts[:,1] <= ymx)):
                    return 1.0
            return 0.0
        line = LineString(pts)
        inter = self.multi_poly.intersection(line)
        return getattr(inter, 'length', 0.0)

    def draw(self, ax):
        for poly in self.poly_list:
            ax.add_patch(patches.Polygon(poly, facecolor='gray', alpha=0.7,
                                         edgecolor='black', linewidth=1))


# =============================================================================
# Online MPC wrapper around VP-STO (+ Terminal-Guard + zero-via latch + idle)
# =============================================================================
class OnlineVPSTOMPC:
    def __init__(self,
                 q_goal: np.ndarray,
                 q_min: np.ndarray,
                 q_max: np.ndarray,
                 env: CollisionEnvironment,
                 *,
                 ndof: int = 2,
                 N_eval: int = 100,
                 N_via: int = 5,
                 pop_size_warm: int = 48,
                 pop_size_explore: int = 96,
                 sigma_warm: float = 0.08,
                 sigma_explore: float = 0.35,  # 建议保持 0.25~0.5，避免近端抖动
                 iters_per_tick_warm: int = 6,
                 iters_per_tick_explore: int = 25,
                 vel_lim: np.ndarray = None,
                 acc_lim: np.ndarray = None,
                 dt: float = 0.05,
                 hard_collision: float = 1e6,
                 hard_limits: float = 1e6,
                 soft_goal: float = 1e3,
                 smooth_w: float = 1e-3,
                 T_stop: float = 0.8):
        self.ndof = ndof
        self.N_eval = N_eval
        self.N_via = N_via
        self.dt = dt
        self.env = env
        self.q_goal = np.array(q_goal)
        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)
        self.vel_lim = np.array(vel_lim) if vel_lim is not None else 0.2*np.ones(ndof)
        self.acc_lim = np.array(acc_lim) if acc_lim is not None else 0.8*np.ones(ndof)

        # cost weights
        self.hard_collision = hard_collision
        self.hard_limits = hard_limits
        self.soft_goal = soft_goal
        self.smooth_w = smooth_w

        # warm/explore knobs
        self.pop_warm = pop_size_warm
        self.pop_explore = pop_size_explore
        self.sig_warm = sigma_warm
        self.sig_explore = sigma_explore
        self.iters_warm = iters_per_tick_warm
        self.iters_explore = iters_per_tick_explore

        # stateful: previous (best) plan for warm start / idle
        self.prev_sol = None   # a VPSTOSolution
        self.prev_T = None

        # trajectory utilities
        self.vptraj = VPTraj(ndof, N_eval, N_via, self.vel_lim, self.acc_lim)
        # zero-via（直连）专用的 VPTraj（N_via=1 → 无内部 via）
        self.vptraj_direct = VPTraj(ndof, N_eval, 1, self.vel_lim, self.acc_lim)
        self.T_stop = T_stop

        # logging for animation
        self.trajectory_log = []   # each tick: best planned trajectory (K, ndof)
        self.candidate_log = []    # each tick: dict {'pos','vel','T'} or None
        self.mode_log = []         # 'zero' | 'idle' | 'warm' | 'exploration'
        self.mode = None
        self.best_via_log = []     # each tick: (N_via-1, ndof) or None
        self.compute_time_log = [] # seconds
        self.T_best_log = []       # each tick: T_best value for visualization

        # -------- IDLE 模式参数与状态 ----------
        self.idle = False
        self._idle_cnt = 0
        self.idle_pos_tol = 3e-2
        self.idle_vel_tol = 3e-2
        self.idle_enter_count = 2
        self.idle_exit_pos = 2.5*self.idle_pos_tol
        self.idle_exit_vel = 3.0*self.idle_vel_tol

        # 终端吸附（snap）
        self.snap_pos_tol = 5e-3
        self.snap_vel_tol = 1e-2

        # -------- TERMINAL GUARD：进入终端小邻域后彻底停住 ----------
        self._terminal_hold = False
        self.terminal_pos_tol = 1e-3    # 例如 0.001 m
        self.terminal_vel_tol = 2e-2
        self.terminal_release_pos = 3*self.terminal_pos_tol
        self.terminal_release_vel = self.idle_exit_vel

        # -------- ZERO LATCH：直连后滚动尾段 ----------
        self._zero_latch = False
        self._direct_traj_pos = None  # (K, ndof)
        self._direct_traj_vel = None  # (K, ndof)
        self._direct_idx = 0          # 当前处于哪一个离散采样点
        self._direct_stride = 1       # Δt 对应的采样步数
        self._direct_T_rem = 0.0      # 剩余时长（近似）

    # ---------- loss builders (vectorized over candidates) ------------------
    def _bounds_violation(self, q: np.ndarray) -> np.ndarray:
        below = (q < self.q_min[None, None, :]).sum(axis=(1, 2))
        above = (q > self.q_max[None, None, :]).sum(axis=(1, 2))
        return below + above

    def _collision_cost(self, q: np.ndarray) -> np.ndarray:
        cost = np.zeros(q.shape[0])
        for i in range(q.shape[0]):
            L = self.env.collision_length(q[i])
            cost[i] = (L > 0.0) * (1.0 + L)
        return cost

    def _smoothness(self, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        v2 = (dq**2).sum(-1)
        a2 = (ddq**2).sum(-1)
        va = (dq*ddq).sum(-1)
        kappa2 = (v2*a2 - va**2) / (np.maximum(v2, 1e-8)**3)
        return kappa2.mean(axis=1)

    def _mpc_loss(self, cand: dict) -> np.ndarray:
        q, dq, ddq, T = cand['pos'], cand['vel'], cand['acc'], cand['T']
        goal_err = np.linalg.norm(q[:, -1, :] - self.q_goal[None, :], axis=1)
        bounds_v = self._bounds_violation(q)
        coll = self._collision_cost(q)
        smooth = self._smoothness(dq, ddq)
        
        # 添加单调性奖励，防止绕圈
        monotonic_penalty = self._monotonic_penalty(q)
        
        return (T
                + self.smooth_w * smooth
                + self.hard_limits * bounds_v
                + self.hard_collision * coll
                + self.soft_goal * goal_err
                + monotonic_penalty)

    def _monotonic_penalty(self, q: np.ndarray) -> np.ndarray:
        """计算路径单调性惩罚，防止绕圈行为"""
        penalties = np.zeros(q.shape[0])
        
        for i in range(q.shape[0]):
            traj = q[i]  # (N_eval, ndof)
            distances = np.linalg.norm(traj - self.q_goal[None, :], axis=1)
            
            # 计算距离的非单调性
            non_monotonic = 0.0
            for j in range(len(distances) - 1):
                if distances[j+1] > distances[j]:  # 距离增加了
                    non_monotonic += (distances[j+1] - distances[j])
            
            # 接近目标时加重惩罚权重
            current_dist = distances[0]
            if current_dist < 0.18:
                weight = 1e7 * (0.18 - current_dist) / 0.18  # 越近惩罚越重
                penalties[i] = weight * non_monotonic
            else:
                penalties[i] = 100.0 * non_monotonic  # 基础惩罚
                
        return penalties

    # ---------- idle helpers -------------------------------------------------
    def _enter_idle(self):
        self.idle = True
        self._idle_cnt = 0

    def _exit_idle(self):
        self.idle = False
        self._idle_cnt = 0

    # ---------- warm start: shift previous plan by Δt -----------------------
    def _time_shifted_p(self, sol, dt: float):
        T = max(float(getattr(sol, 'T_best', 0.0)), 1e-3)
        if T <= dt:
            return None
        
        # 如果剩余时间太短，不使用warm start
        if T - dt < 2.0 * dt:
            return None
            
        t_grid = np.linspace(dt, T, self.N_via + 1)
        q_grid, _, _ = sol.get_posvelacc(t_grid)
        p_init = q_grid[1:-1].reshape(-1)
        
        # 边界检查：确保via points在合理范围内
        via_pts = p_init.reshape(self.N_via-1, self.ndof)
        
        # 检查是否有via point超出边界
        out_of_bounds = np.any((via_pts < self.q_min) | (via_pts > self.q_max))
        if out_of_bounds:
            return None
            
        # 检查via points是否合理地接近目标（单调性检查）
        current_dist = float(np.linalg.norm(q_grid[0] - self.q_goal))  # 当前位置到目标距离
        
        for i, via_pt in enumerate(via_pts):
            via_dist = float(np.linalg.norm(via_pt - self.q_goal))
            # 如果via point距离目标比当前位置还远，可能不合理
            if via_dist > current_dist * 1.5:  # 允许一定的偏差
                return None
        
        return p_init

    # ---------- zero-via direct try -----------------------------------------
    def _try_direct(self, q: np.ndarray, dq: np.ndarray, T_stop_local: float):
        """Return (success, qdt, dqdt, q_traj, dq_traj, T0, stride) for zero-via direct plan."""
        P0 = np.zeros((1, 0))
        dqT = np.zeros(self.ndof)
        T0 = float(self.vptraj_direct.get_min_duration(P0, q0=q, dq0=dq, qT=self.q_goal, dqT=dqT))
        if T0 <= 0:
            return False, None, None, None, None, None, None

        q_traj, dq_traj, _ = self.vptraj_direct.get_trajectory(P0, q0=q, dq0=dq, qT=self.q_goal, dqT=dqT, T=T0)
        q_traj, dq_traj = q_traj[0], dq_traj[0]

        # bound / collision
        if self._bounds_violation(q_traj[None, ...])[0] > 0:
            return False, None, None, None, None, None, None
        if self._collision_cost(q_traj[None, ...])[0] > 0:
            return False, None, None, None, None, None, None

        # 允许时长（动态阈值，带下限）
        if T0 > T_stop_local:
            return False, None, None, None, None, None, None

        # snap if extremely close
        if float(np.linalg.norm(q - self.q_goal)) < self.snap_pos_tol:
            qdt = self.q_goal.copy()
            dqdt = np.zeros_like(qdt)
            stride = max(1, int(round(self.dt / T0 * (self.N_eval - 1))))
            return True, qdt, dqdt, q_traj, dq_traj, T0, stride

        # Δt 对应的索引步长
        stride = max(1, int(round(self.dt / T0 * (self.N_eval - 1))))
        k = stride
        qdt = q_traj[k].copy()
        dqdt = dq_traj[k].copy()
        return True, qdt, dqdt, q_traj, dq_traj, T0, stride

    # ---------- roll the latched direct trajectory --------------------------
    def _roll_zero_latch(self, tic):
        """Use stored direct trajectory to roll one Δt forward without any sampling."""
        if (self._direct_traj_pos is None) or (self._direct_traj_vel is None):
            self._zero_latch = False
            return None

        N = self._direct_traj_pos.shape[0]
        if (self._direct_idx >= N - 1) or (self._direct_T_rem <= 0.0):
            # 直连尾段结束 → 若已经非常近，进入 Terminal-Hold
            dist_tail = float(np.linalg.norm(self._direct_traj_pos[-1] - self.q_goal))
            if dist_tail <= self.terminal_pos_tol:
                self._terminal_hold = True
                qdt = self.q_goal.copy()
                dqdt = np.zeros_like(qdt)
                self.trajectory_log.append(self._direct_traj_pos)
                self.candidate_log.append(None)
                self.best_via_log.append(None)
                self.mode = 'idle'
                self.mode_log.append('idle')
                self.T_best_log.append(0.0)  # Direct trajectory completed
                self.compute_time_log.append(time.perf_counter()-tic)
                self._zero_latch = False
                return qdt, dqdt
            self._zero_latch = False
            return None

        nxt = min(self._direct_idx + self._direct_stride, N - 1)
        qdt = self._direct_traj_pos[nxt].copy()
        dqdt = self._direct_traj_vel[nxt].copy()
        self._direct_idx = nxt
        self._direct_T_rem -= self.dt

        # 日志：计划线仍画整条（更直观）
        self.trajectory_log.append(self._direct_traj_pos)
        self.candidate_log.append(None)
        self.best_via_log.append(None)

        # 模式标签：靠近就显示 idle，否则 zero；靠近则置位终端锁定
        if (np.linalg.norm(qdt - self.q_goal) < self.idle_pos_tol) and (np.linalg.norm(dqdt) < self.idle_vel_tol):
            mode = 'idle'
            self._terminal_hold = True
        else:
            mode = 'zero'
        self.mode = mode
        self.mode_log.append(mode)
        self.T_best_log.append(self._direct_T_rem)  # Remaining direct trajectory time
        self.compute_time_log.append(time.perf_counter() - tic)
        return qdt, dqdt

    # ---------- one MPC tick -----------------------------------------------
    def step(self, q: np.ndarray, dq: np.ndarray):
        tic = time.perf_counter()

        # shape guard
        q = np.asarray(q, dtype=float).reshape(self.ndof)
        dq = np.asarray(dq, dtype=float).reshape(self.ndof)
        self.q_goal = np.asarray(self.q_goal, dtype=float).reshape(self.ndof)

        # ---------- Terminal Guard（优先级 #1） ----------
        dist_now = float(np.linalg.norm(q - self.q_goal))
        vel_now  = float(np.linalg.norm(dq))
        if self._terminal_hold:
            if (dist_now <= self.terminal_release_pos) and (vel_now <= self.terminal_release_vel):
                qdt = self.q_goal.copy()
                dqdt = np.zeros_like(qdt)
                # 为了动画友好，画一条短线（当前→目标）
                self.trajectory_log.append(np.vstack([q, qdt]))
                self.candidate_log.append(None)
                self.best_via_log.append(None)
                self.mode = 'idle'
                self.mode_log.append('idle')
                self.T_best_log.append(0.0)  # Terminal hold, no planning
                self.compute_time_log.append(time.perf_counter()-tic)
                return qdt, dqdt
            else:
                self._terminal_hold = False  # 被拖离，解除终端锁定

        # ---------- Zero-Latch（优先级 #2） ----------
        if self._zero_latch:
            rolled = self._roll_zero_latch(tic)
            if rolled is not None:
                return rolled  # (qdt, dqdt)

        # ---------- Idle-Tail of VPSTO（优先级 #3） ----------
        mode_warmstart = (self.prev_sol is not None) and (self.prev_T is not None) and (self.prev_T > 2.0*self.dt)
        if self.idle:
            if (np.linalg.norm(q - self.q_goal) > self.idle_exit_pos) or \
               (np.linalg.norm(dq) > self.idle_exit_vel) or \
               (self.prev_sol is None) or (self.prev_T is None) or (self.prev_T <= 0.0):
                self._exit_idle()
            else:
                qdt, dqdt, _ = self.prev_sol.get_posvelacc(self.dt)
                qdt, dqdt = qdt[0], dqdt[0]
                self.prev_T -= self.dt
                # log remainder
                try:
                    t_full = np.linspace(0.0, max(self.prev_T, self.dt), self.N_eval)
                    q_traj, _, _ = self.prev_sol.get_posvelacc(t_full)
                    self.trajectory_log.append(q_traj)
                except Exception:
                    self.trajectory_log.append(None)
                self.candidate_log.append(None)
                self.mode = 'idle'
                self.mode_log.append('idle')
                self.best_via_log.append(None)
                # 极近时进入终端锁定
                if (np.linalg.norm(qdt - self.q_goal) < self.terminal_pos_tol) and (np.linalg.norm(dqdt) < self.terminal_vel_tol):
                    self._terminal_hold = True
                self.T_best_log.append(max(self.prev_T, 0.0))  # Remaining idle trajectory time
                self.compute_time_log.append(time.perf_counter()-tic)
                return qdt, dqdt

        # ---------- Zero-Try（优先级 #4） ----------
        vmax = float(np.linalg.norm(self.vel_lim)) + 1e-9
        T_stop_local = max(2.0*self.dt, min(self.T_stop, 2.5 * dist_now / vmax))
        ok, qdt_d, dqdt_d, qtraj_d, dqtraj_d, T0, stride = self._try_direct(q, dq, T_stop_local)
        if ok:
            # 记录/锁定直连，下一拍开始滚动尾段（本拍仍返回 zero）
            self.trajectory_log.append(qtraj_d if qtraj_d is not None else None)
            self.candidate_log.append(None)
            self.mode = 'zero'
            self.mode_log.append('zero')
            self.best_via_log.append(None)
            # 直连锁定状态
            if (qtraj_d is not None) and (dqtraj_d is not None) and (T0 is not None) and (stride is not None):
                self._zero_latch = True
                self._direct_traj_pos = qtraj_d
                self._direct_traj_vel = dqtraj_d
                self._direct_stride   = max(1, int(stride))
                self._direct_idx      = self._direct_stride  # 本拍已经走了 Δt
                self._direct_T_rem    = max(0.0, T0 - self.dt)
            else:
                self._zero_latch = False
            # 若本拍已非常近，则直接锁定终端
            if (np.linalg.norm(qdt_d - self.q_goal) < self.idle_pos_tol) and (np.linalg.norm(dqdt_d) < self.idle_vel_tol):
                self._terminal_hold = True
            self.T_best_log.append(T0 if T0 is not None else 0.0)  # Direct trajectory time
            self.compute_time_log.append(time.perf_counter()-tic)
            return qdt_d, dqdt_d

        # ---------- VP-STO（优先级 #5） ----------
        mode_label = 'warm' if mode_warmstart else 'exploration'

        opt = VPSTOOptions(self.ndof)
        opt.vel_lim = self.vel_lim.copy()
        opt.acc_lim = self.acc_lim.copy()
        opt.N_eval = self.N_eval
        opt.N_via = self.N_via
        opt.CMA_diagonal = True
        opt.pop_size = self.pop_warm if mode_warmstart else self.pop_explore
        
        # 自适应sigma：接近目标时减少探索范围
        base_sigma_warm = self.sig_warm
        base_sigma_explore = self.sig_explore
        
        if dist_now < 0.15:  # 接近目标时
            distance_factor = max(0.3, dist_now / 0.15)  # 距离因子：0.3到1.0
            opt.sigma_init = (base_sigma_warm if mode_warmstart else base_sigma_explore) * distance_factor
        else:
            opt.sigma_init = base_sigma_warm if mode_warmstart else base_sigma_explore
            
        opt.max_iter = self.iters_warm if mode_warmstart else self.iters_explore
        opt.log = False
        opt.verbose = False

        # DEBUG: 添加详细的调试信息
        if dist_now < 0.15:  # 扩大调试范围
            print(f"  Near goal: {dist_now:.4f} m, vel {vel_now:.4f} m/s, using VP-STO {mode_label}")
            print(f"    Parameters: pop {opt.pop_size}, sigma {opt.sigma_init:.4f} (base: {base_sigma_warm if mode_warmstart else base_sigma_explore:.4f}), iters {opt.max_iter}")
            if mode_warmstart and self.prev_sol is not None:
                print(f"    Prev solution T: {getattr(self.prev_sol, 'T_best', 'N/A'):.4f}")
                print(f"    Remaining T: {self.prev_T:.4f}")
        solver = VPSTO(opt)
        if mode_warmstart:
            p_init = self._time_shifted_p(self.prev_sol, self.dt)
            if p_init is not None and p_init.size == self.ndof*(self.N_via-1):
                solver.set_initial_guess(p_init)
                # DEBUG: 分析warm start的via points
                if dist_now < 0.15:
                    via_pts = p_init.reshape(self.N_via-1, self.ndof)
                    print(f"    Warm start via points:")
                    for i, pt in enumerate(via_pts):
                        dist_to_goal = np.linalg.norm(pt - self.q_goal)
                        print(f"      Via {i}: [{pt[0]:.4f}, {pt[1]:.4f}], dist_to_goal: {dist_to_goal:.4f}")
            else:
                if dist_now < 0.15:
                    print(f"    Warm start failed: p_init is None or wrong size")

        dqT = np.zeros(self.ndof, dtype=float)
        sol = solver.minimize(self._mpc_loss, q0=q, dq0=dq, qT=self.q_goal, dqT=dqT, T=None)
        Tbest = float(getattr(sol, 'T_best', 0.0))

        # DEBUG: 分析最终解
        if dist_now < 0.15:
            print(f"    Solution T: {Tbest:.4f}")
            if hasattr(sol, 'p_best') and sol.p_best is not None:
                via_best_debug = sol.p_best.reshape(self.N_via-1, self.ndof)
                print(f"    Solution via points:")
                for i, pt in enumerate(via_best_debug):
                    dist_to_goal = np.linalg.norm(pt - self.q_goal)
                    print(f"      Via {i}: [{pt[0]:.4f}, {pt[1]:.4f}], dist_to_goal: {dist_to_goal:.4f}")
                
                # 检查路径是否合理（单调接近目标）
                current_pos = q.copy()
                distances = [np.linalg.norm(current_pos - self.q_goal)]
                for pt in via_best_debug:
                    distances.append(np.linalg.norm(pt - self.q_goal))
                distances.append(0.0)  # 最终到达目标
                
                # 检查距离是否单调递减
                is_monotonic = all(distances[i] >= distances[i+1] for i in range(len(distances)-1))
                print(f"    Path distances: {[f'{d:.4f}' for d in distances]}")
                print(f"    Is monotonic approaching: {is_monotonic}")
                
                if not is_monotonic:
                    print(f"    WARNING: Non-monotonic path detected!")

        # candidates for animation (last generation if exposed)
        if hasattr(sol, 'candidates') and isinstance(sol.candidates, dict) and sol.candidates.get('pos', None) is not None:
            try:
                self.candidate_log.append({
                    'pos': sol.candidates['pos'].copy(),
                    'vel': sol.candidates['vel'].copy(),
                    'T':   sol.candidates['T'].copy()
                })
            except Exception:
                self.candidate_log.append(None)
        else:
            self.candidate_log.append(None)

        # visualize the sol.candidates['T]
        # fig, ax = plt.subplots(figsize=(14, 14))
        # plt.plot(sol.candidates['T'], label='Candidate Times')
        # plt.scatter(range(len(sol.candidates['T'])), sol.candidates['T'], color='red', marker='o', s=100)
        # plt.xlabel('Candidate Index')
        # plt.ylabel('Time (s)')
        # plt.title('Distribution of Candidate Times')
        # plt.legend()
        # plt.show()
        # via points of best (if available)
        via_best = None
        try:
            if hasattr(sol, 'p_best') and sol.p_best is not None and self.N_via > 1:
                via_best = sol.p_best.reshape(self.N_via-1, self.ndof).copy()
        except Exception:
            via_best = None

        if Tbest <= 0:
            self.trajectory_log.append(None)
            self.mode = mode_label
            self.mode_log.append(mode_label)
            self.best_via_log.append(via_best)
            self.T_best_log.append(0.0)  # Failed optimization
            self.compute_time_log.append(time.perf_counter()-tic)
            return q.copy(), np.zeros_like(dq)

        if Tbest <= self.dt:
            q_next, dq_next, _ = sol.get_posvelacc(Tbest)
            q_next = q_next[0]
            dq_next = np.zeros_like(q_next)
            try:
                t_full = np.linspace(0, Tbest, self.N_eval)
                q_traj, _, _ = sol.get_posvelacc(t_full)
                self.trajectory_log.append(q_traj)
            except Exception:
                self.trajectory_log.append(None)
            self.prev_sol = None
            self.prev_T = None
            self.mode = mode_label
            self.mode_log.append(mode_label)
            self.best_via_log.append(via_best)
            self.T_best_log.append(Tbest)  # Short trajectory completed
            self.compute_time_log.append(time.perf_counter()-tic)
            return q_next, dq_next

        # short-horizon extraction
        qdt, dqdt, _ = sol.get_posvelacc(self.dt)
        qdt, dqdt = qdt[0], dqdt[0]

        # 终端吸附
        if (np.linalg.norm(qdt - self.q_goal) < self.snap_pos_tol) and (np.linalg.norm(dqdt) < self.snap_vel_tol):
            qdt = self.q_goal.copy()
            dqdt = np.zeros_like(qdt)

        self.prev_sol = sol
        self.prev_T = Tbest - self.dt

        # log full best plan
        try:
            t_full = np.linspace(0, Tbest, self.N_eval)
            q_traj, _, _ = sol.get_posvelacc(t_full)
            self.trajectory_log.append(q_traj)
        except Exception:
            self.trajectory_log.append(None)

        # idle enter logic（更快）
        if (np.linalg.norm(qdt - self.q_goal) < self.idle_pos_tol) and (np.linalg.norm(dqdt) < self.idle_vel_tol):
            self._idle_cnt += 1
        else:
            self._idle_cnt = 0
        if (not self.idle) and (self._idle_cnt >= self.idle_enter_count):
            self._enter_idle()

        # 极近时锁定终端（下一拍由 Terminal-Guard 接管）
        if (np.linalg.norm(qdt - self.q_goal) < self.terminal_pos_tol) and (np.linalg.norm(dqdt) < self.terminal_vel_tol):
            self._terminal_hold = True

        self.mode = mode_label
        self.mode_log.append(mode_label)
        self.best_via_log.append(via_best)
        self.T_best_log.append(Tbest)  # Current best trajectory time
        self.compute_time_log.append(time.perf_counter()-tic)
        return qdt, dqdt


# =============================================================================
# Demo runner
# =============================================================================
def run_demo():
    q_min = 0.0*np.ones(2)
    q_max = 0.5*np.ones(2)
    env = CollisionEnvironment()

    # q0 = np.array([0.1, 0.25])
    # dq0 = np.zeros(2)
    # q_goal = np.array([0.35, 0.25])

    # setup from 02.py
    q0 = np.array([0.1, 0.1])
    dq0 = np.zeros(2)
    q_goal = np.array([0.4, 0.1])

    ctrl = OnlineVPSTOMPC(
        q_goal=q_goal,
        q_min=q_min,
        q_max=q_max,
        env=env,
        ndof=2,
        N_eval=100,
        N_via=5,
        pop_size_warm=48,
        pop_size_explore=96,
        sigma_warm=0.08,
        sigma_explore=5.35,  
        iters_per_tick_warm=6,
        iters_per_tick_explore=18,
        vel_lim=np.array([0.10, 0.10]),
        acc_lim=np.array([0.50, 0.50]),
        dt=0.05,
        hard_collision=1e6,
        hard_limits=1e6,
        soft_goal=1e3,
        smooth_w=1e-3,
        T_stop=0.8,
    )

    # perfect tracking sim (like an impedance layer would do)
    Tsim = 12.0
    steps = int(Tsim/ctrl.dt)
    q_hist = [q0.copy()]
    dq_hist = [dq0.copy()]
    ddq_hist = [np.zeros(2)]  # 加速度历史记录

    time_start = time.perf_counter()
    for i in range(steps):
        q_ref, dq_ref = ctrl.step(q_hist[-1], dq_hist[-1])
        
        # 计算加速度（数值微分）
        if i > 0:
            ddq_current = (dq_ref - dq_hist[-1]) / ctrl.dt
        else:
            ddq_current = np.zeros(2)
        
        q_hist.append(q_ref.copy())
        dq_hist.append(dq_ref.copy())
        ddq_hist.append(ddq_current.copy())
        
        if np.linalg.norm(q_hist[-1] - q_goal) < 1e-2 and np.linalg.norm(dq_hist[-1]) < 1e-2:
            break
    time_end = time.perf_counter()
    q_hist = np.array(q_hist)
    dq_hist = np.array(dq_hist)
    ddq_hist = np.array(ddq_hist)
    t = np.arange(len(q_hist))*ctrl.dt

    # --- plots --------------------------------------------------------------
    print(f"Simulation took {time_end - time_start:.2f}s, {len(q_hist)} steps.")
    print("Creating plots...")
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    # path
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlim(0, q_max[0])
    ax0.set_ylim(0, q_max[1])
    ax0.grid(True, alpha=0.3)
    env.draw(ax0)
    ax0.plot(q_hist[:, 0], q_hist[:, 1], 'b-', lw=2, label='MPC path')
    ax0.scatter(q_hist[0, 0], q_hist[0, 1], c='green', s=80, marker='o', label='Start')
    ax0.scatter(q_goal[0], q_goal[1], c='red', s=120, marker='*', label='Goal')
    ax0.legend(loc='lower right')
    ax0.set_title('VP-STO MPC (Terminal-Guard + zero-latch + idle-tail)')

    # velocity
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, dq_hist[:, 0], label='vx')
    ax1.plot(t, dq_hist[:, 1], label='vy')
    ax1.axhline(+ctrl.vel_lim[0], ls='--', c='r', lw=1)
    ax1.axhline(-ctrl.vel_lim[0], ls='--', c='r', lw=1)
    ax1.set_ylabel('velocity (m/s)')
    ax1.set_title('Velocity Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # acceleration
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t, ddq_hist[:, 0], label='ax')
    ax2.plot(t, ddq_hist[:, 1], label='ay')
    ax2.axhline(+ctrl.acc_lim[0], ls='--', c='r', lw=1)
    ax2.axhline(-ctrl.acc_lim[0], ls='--', c='r', lw=1)
    ax2.set_ylabel('acceleration (m/s²)')
    ax2.set_xlabel('time (s)')
    ax2.set_title('Acceleration Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # distance to goal
    ax3 = fig.add_subplot(gs[:, 2])
    dist = np.linalg.norm(q_hist - q_goal[None, :], axis=1)
    ax3.plot(t, dist)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('‖q - q_goal‖')
    ax3.set_title('Distance to Goal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Finished in {t[-1]:.2f}s, final error {dist[-1]:.4f}")

    # Create animation with optional saving
    print("Creating animation...")
    save_path = None
    if SAVE_ANIMATION:
        save_path = ANIMATION_FILENAME
        print(f"Animation will be automatically saved as: {save_path}")
    else:
        # Ask user if they want to save the animation
        try:
            save_animation = input("Do you want to save the animation? (y/n): ").lower().strip()
            if save_animation in ['y', 'yes']:
                save_filename = input("Enter filename (supports .mp4 or .gif, default: mpc_animation.mp4): ").strip()
                if not save_filename:
                    save_filename = "mpc_animation.mp4"
                if not (save_filename.lower().endswith('.mp4') or save_filename.lower().endswith('.gif')):
                    save_filename += '.mp4'
                save_path = save_filename
                print(f"Animation will be saved as: {save_path}")
        except (EOFError, KeyboardInterrupt):
            print("Running in non-interactive mode, animation will not be saved.")
    create_animation(q_hist, dq_hist, ddq_hist, ctrl, env, t, q0, q_goal, q_min, q_max, save_path)


def create_animation(q_hist, dq_hist, ddq_hist, ctrl, env, time_vec, q0, q_goal, q_min, q_max, save_path=None):
    """Create and display animation of the MPC trajectory

    Args:
        q_hist: Position history
        dq_hist: Velocity history  
        ddq_hist: Acceleration history
        save_path (str, optional): Path to save the animation. If provided, will save as MP4 or GIF.
    """
    robot_radius = 0.006

    # 模式对应的颜色（planned_line）
    MODE_COLORS = {
        'zero':        'tab:green',
        'idle':        'tab:gray',
        'warm':        'tab:orange',
        'exploration': 'tab:purple',
        None:          'tab:purple'  # fallback
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    # Left subplot: trajectory animation
    ax1.set_facecolor('white')
    ax1.set_xlim(0 - 0.02, q_max[0] + 0.02)
    ax1.set_ylim(0 - 0.02, q_max[1] + 0.02)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('VP-STO MPC Real-time Trajectory')
    ax1.set_aspect('equal', adjustable='box')

    env.draw(ax1)

    robot_patch = plt.Circle(q0, robot_radius, color='blue', alpha=0.7, zorder=10)
    ax1.add_patch(robot_patch)

    ax1.scatter(q0[0], q0[1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(q_goal[0], q_goal[1], c='red', s=120, marker='*', label='Goal', zorder=5)

    executed_line, = ax1.plot([], [], color='tab:blue', linewidth=3, alpha=0.9, label='Executed path')
    planned_line,  = ax1.plot([], [], color='tab:purple', linewidth=2.5, alpha=0.9, label='Current best')

    # candidate trajectories (last generation)
    max_candidates = max(ctrl.pop_warm, ctrl.pop_explore)
    candidate_lines = []
    for _ in range(min(max_candidates, 50)):
        ln, = ax1.plot([], [], color='orange', alpha=0.25, linewidth=0.8)
        candidate_lines.append(ln)

    # via points (best of the tick; hidden for zero mode)
    via_scat = ax1.scatter([], [], c='k', marker='x', s=35, zorder=6, label='via points')

    # Text overlays
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12,
                         va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    status_text = ax1.text(0.02, 0.15, '', transform=ax1.transAxes, fontsize=10,
                           va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Right subplot: velocity profiles
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(+ctrl.vel_lim[0], ls='--', c='r', lw=1, alpha=0.7, label='Velocity limits')
    ax2.axhline(-ctrl.vel_lim[0], ls='--', c='r', lw=1, alpha=0.7)
    vx_line, = ax2.plot([], [], 'b-', label='vx', linewidth=2)
    vy_line, = ax2.plot([], [], 'r-', label='vy', linewidth=2)
    current_time_line_v = ax2.axvline(0, color='black', linestyle='-', alpha=0.7, label='Current time')
    ax2.legend()

    # Far right subplot: acceleration profiles
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Acceleration Profile')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(+ctrl.acc_lim[0], ls='--', c='r', lw=1, alpha=0.7, label='Acceleration limits')
    ax3.axhline(-ctrl.acc_lim[0], ls='--', c='r', lw=1, alpha=0.7)
    ax_line, = ax3.plot([], [], 'b-', label='ax', linewidth=2)
    ay_line, = ax3.plot([], [], 'r-', label='ay', linewidth=2)
    current_time_line_a = ax3.axvline(0, color='black', linestyle='-', alpha=0.7, label='Current time')
    ax3.legend()

    def init():
        executed_line.set_data([], [])
        planned_line.set_data([], [])
        for ln in candidate_lines:
            ln.set_data([], [])
        via_scat.set_offsets(np.empty((0,2)))
        vx_line.set_data([], [])
        vy_line.set_data([], [])
        ax_line.set_data([], [])
        ay_line.set_data([], [])
        time_text.set_text('')
        status_text.set_text('')
        planned_line.set_color(MODE_COLORS[None])
        return [executed_line, planned_line, *candidate_lines, via_scat,
                vx_line, vy_line, ax_line, ay_line, robot_patch, time_text, status_text, 
                current_time_line_v, current_time_line_a]

    def animate(frame):
        current_step = min(frame, len(q_hist) - 1)
        current_time = time_vec[current_step] if current_step < len(time_vec) else time_vec[-1]

        robot_patch.center = q_hist[current_step]

        if current_step > 0:
            executed_line.set_data(q_hist[:current_step+1, 0], q_hist[:current_step+1, 1])

        # best plan (this tick)
        if current_step < len(ctrl.trajectory_log) and ctrl.trajectory_log[current_step] is not None:
            planned_traj = ctrl.trajectory_log[current_step]
            planned_line.set_data(planned_traj[:, 0], planned_traj[:, 1])
        else:
            planned_line.set_data([], [])

        # candidates (last generation)
        for ln in candidate_lines:
            ln.set_data([], [])
        if current_step < len(ctrl.candidate_log) and ctrl.candidate_log[current_step] is not None:
            cand = ctrl.candidate_log[current_step]
            P = cand['pos']
            n_show = min(len(candidate_lines), P.shape[0], 50)
            for i in range(n_show):
                candidate_lines[i].set_data(P[i, :, 0], P[i, :, 1])

        # via points (best of tick), skip if zero mode
        if current_step < len(ctrl.mode_log):
            mode = ctrl.mode_log[current_step]
        else:
            mode = None
        if (mode == 'zero') or (current_step >= len(ctrl.best_via_log)) or (ctrl.best_via_log[current_step] is None):
            via_scat.set_offsets(np.empty((0,2)))
        else:
            via_pts = ctrl.best_via_log[current_step]
            if via_pts.size == 0:
                via_scat.set_offsets(np.empty((0,2)))
            else:
                via_scat.set_offsets(via_pts[:, :2])

        # velocity panel
        if current_step > 0:
            vx_line.set_data(time_vec[:current_step+1], dq_hist[:current_step+1, 0])
            vy_line.set_data(time_vec[:current_step+1], dq_hist[:current_step+1, 1])
            ax2.set_xlim(0, max(time_vec[current_step] + 1, 2))
            max_vel = max(np.abs(dq_hist[:current_step+1]).max(), ctrl.vel_lim[0]) * 1.1
            ax2.set_ylim(-max_vel, max_vel)

        # acceleration panel
        if current_step > 0:
            ax_line.set_data(time_vec[:current_step+1], ddq_hist[:current_step+1, 0])
            ay_line.set_data(time_vec[:current_step+1], ddq_hist[:current_step+1, 1])
            ax3.set_xlim(0, max(time_vec[current_step] + 1, 2))
            max_acc = max(np.abs(ddq_hist[:current_step+1]).max(), ctrl.acc_lim[0]) * 1.1
            ax3.set_ylim(-max_acc, max_acc)

        current_time_line_v.set_xdata([current_time, current_time])
        current_time_line_a.set_xdata([current_time, current_time])

        # planned color by mode
        planned_line.set_color(MODE_COLORS.get(mode, MODE_COLORS[None]))

        # compute-time stats
        if current_step < len(ctrl.compute_time_log):
            cur_ms = ctrl.compute_time_log[current_step] * 1000.0
            cur_hz = (1.0 / ctrl.compute_time_log[current_step]) if ctrl.compute_time_log[current_step] > 1e-9 else float('inf')
            avg_ms = np.mean(ctrl.compute_time_log[:current_step+1]) * 1000.0
            avg_hz = 1.0 / (avg_ms / 1000.0) if avg_ms > 1e-9 else float('inf')
        else:
            cur_ms = avg_ms = 0.0
            cur_hz = avg_hz = 0.0

        # mode freq (up to current_step)
        total = current_step + 1
        def pct(label):
            return int(round(100.0 * ctrl.mode_log[:current_step+1].count(label) / total)) if total > 0 else 0
        zf, wf, ef, ifq = pct('zero'), pct('warm'), pct('exploration'), pct('idle')

        # overlays
        speed = np.linalg.norm(dq_hist[current_step]) if current_step < len(dq_hist) else 0.0
        distance_to_goal = np.linalg.norm(q_hist[current_step] - q_goal)
        
        # Get current T_best
        current_T_best = ctrl.T_best_log[current_step] if current_step < len(ctrl.T_best_log) else 0.0

        time_text.set_text(f'Time: {current_time:.2f}s\nStep: {current_step}/{len(q_hist)-1}')
        status_text.set_text(
            f'Mode: {mode}\n'
            f'Speed: {speed:.3f}\nDist→Goal: {distance_to_goal:.4f}\n'
            f'T_best: {current_T_best:.4f}s\n'
            f'Compute: cur {cur_ms:.1f} ms ({cur_hz:.1f} Hz) | avg {avg_ms:.1f} ms ({avg_hz:.1f} Hz)\n'
            f'Freq: Z {zf}% | W {wf}% | E {ef}% | I {ifq}%'
        )

        return [executed_line, planned_line, *candidate_lines, via_scat,
                vx_line, vy_line, ax_line, ay_line, robot_patch, time_text, status_text, 
                current_time_line_v, current_time_line_a]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(q_hist), interval=50, blit=False, repeat=True)

    # Save animation if path is provided
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        try:
            if save_path.lower().endswith('.gif'):
                # Save as GIF
                print("Saving as GIF format...")
                Writer = animation.writers['pillow']
                writer = Writer(fps=15)
                anim.save(save_path, writer=writer)
            else:
                # Save as MP4 (default)
                print("Saving as MP4 format...")
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=20, metadata=dict(artist='VP-STO MPC'), bitrate=1800)
                anim.save(save_path, writer=writer)
            print(f"Animation saved successfully to {save_path}")
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                print(f"File size: {file_size:.2f} MB")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            if save_path.lower().endswith('.mp4'):
                print("Make sure ffmpeg is installed: sudo apt-get install ffmpeg")
            elif save_path.lower().endswith('.gif'):
                print("Make sure pillow is installed: pip install pillow")

    plt.tight_layout()
    plt.show()
    return anim


if __name__ == "__main__":
    run_demo()
