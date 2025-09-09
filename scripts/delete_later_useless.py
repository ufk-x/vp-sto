"""
2-DOF Planar Arm VP-STO MPC with Task-Space Polygon Obstacles

- Obstacles live in TASK SPACE (polygons).
- The arm (L1=L2=1 by default) is approximated by many small balls along the two links.
- Collision loss is computed in task space via (unsigned) SDF to polygons; penetration uses a squared hinge.
- Goal is an end-effector (task-space) target x_goal; internally we compute a joint goal q_goal via analytic IK
  chosen to be closest to the current configuration (keeps the VP-STO interface and warm-start dimension stable).
- Animation shows arm, end-effector path, planned/candidate paths, and MPC mode colorization.

Optional: if shapely is installed, SDF (point-to-polygon) is exact & fast; otherwise a pure-numpy fallback is used.
"""

import os
import sys
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

# ================= Animation saving options =================
SAVE_ANIMATION = False
ANIMATION_FILENAME = "mpc_arm_animation.mp4"  # .mp4 or .gif

# --- optional project path (keeps compatibility with your repo layout)
try:
    from config import PLANNER_PATH
    if PLANNER_PATH not in sys.path:
        sys.path.append(PLANNER_PATH)
except Exception:
    pass

# --- VP-STO core -------------------------------------------------------------
from planners.vpsto import VPSTO, VPSTOOptions
from planners.vptraj import VPTraj

# --- shapely (if available) for exact polygon distance & containment ---------
try:
    from shapely.geometry import Polygon, MultiPolygon, Point
except Exception:
    Polygon = MultiPolygon = Point = None


# =============================================================================
# Polygon environment (task space)
# =============================================================================
class TaskSpaceEnvironment:
    """
    Obstacles: list of simple polygons (2D). Provides:
      - draw(ax)
      - distance(points): unsigned distance to the union of polygons, vectorized over points (N,2)
                          returns (N,) np.ndarray
    If shapely is available → use MultiPolygon.distance + .contains for inside=0.
    Else → numpy fallback: min distance to polygon edges; if inside (ray casting) → distance=0.
    """
    def __init__(self, poly_list):
        self.poly_list = [np.asarray(p, float) for p in poly_list]
        if Polygon is not None:
            self.multi_poly = MultiPolygon([Polygon(p) for p in self.poly_list])
        else:
            self.multi_poly = None
            # prepare edges for fast distance
            self.edge_lists = []
            for P in self.poly_list:
                A = P
                B = np.roll(P, -1, axis=0)
                self.edge_lists.append((A, B))

    def draw(self, ax, facecolor='gray', alpha=0.7):
        for poly in self.poly_list:
            ax.add_patch(patches.Polygon(poly, facecolor=facecolor, alpha=alpha,
                                         edgecolor='black', linewidth=1))

    # --- vectorized point-in-polygon (ray casting)
    @staticmethod
    def _point_in_poly(points, poly):
        # points: (N,2), poly: (M,2)
        x = points[:, 0]; y = points[:, 1]
        x0 = poly[:, 0]; y0 = poly[:, 1]
        x1 = np.roll(x0, -1); y1 = np.roll(y0, -1)
        # edges from (x0,y0) to (x1,y1)
        # check if y is between y0 and y1, and compute x-intersect
        cond = ((y0 <= y) & (y < y1)) | ((y1 <= y) & (y < y0))
        x_int = x0 + (y - y0) * (x1 - x0) / np.where((y1 - y0) != 0, (y1 - y0), 1e-12)
        crossings = cond & (x < x_int)
        # xor-reduce along edges
        inside = np.zeros(points.shape[0], dtype=bool)
        # We need per-edge evaluation; do loop over edges (M)
        for i in range(len(poly)):
            yi0, yi1 = y0[i], y1[i]
            ci = (((yi0 <= y) & (y < yi1)) | ((yi1 <= y) & (y < yi0)))
            xi = x0[i] + (y - yi0) * (x1[i] - x0[i]) / ( (y1[i] - yi0) if abs(y1[i]-yi0)>1e-12 else 1e-12 )
            inside ^= (ci & (x < xi))
        return inside

    @staticmethod
    def _dist_points_to_segments(points, A, B):
        # points: (N,2), segments A->B with A:(M,2), B:(M,2)
        # returns min distance over M segments for each point → (N,)
        # Vectorize by broadcasting: compute projection for each segment
        # But avoid huge memory; do chunking
        N = points.shape[0]
        M = A.shape[0]
        out = np.full(N, np.inf, dtype=float)
        CHUNK = 4096
        for s in range(0, N, CHUNK):
            e = min(N, s+CHUNK)
            P = points[s:e, None, :]  # (e-s,1,2)
            A_ = A[None, :, :]        # (1,M,2)
            B_ = B[None, :, :]
            AB = B_ - A_
            AP = P - A_
            t = np.sum(AP*AB, axis=2) / np.clip(np.sum(AB*AB, axis=2), 1e-12, None)  # (e-s,M)
            t = np.clip(t, 0.0, 1.0)
            proj = A_ + t[..., None]*AB   # (e-s,M,2)
            d2 = np.sum((P - proj)**2, axis=2)  # (e-s,M)
            out[s:e] = np.sqrt(np.min(d2, axis=1))
        return out

    def distance(self, pts: np.ndarray, signed: bool = True, chunk_size: int = 50000) -> np.ndarray:
        """
        Vectorized signed distance from many points to the obstacle set.
        pts: (N,2) array
        returns: (N,) signed distance (negative inside or on boundary)
        """
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)

        # --- fast path with Shapely 2.x vectorization ---
        if self.multi_poly is not None:
            try:
                import shapely
                from shapely import points as shp_points
                from shapely import distance as shp_distance
                from shapely import covers as shp_covers  # boundary算inside

                out = np.empty((pts.shape[0],), dtype=float)

                # 巨量点时分块，避免一次性创建超大几何数组/显存峰值
                n = pts.shape[0]
                for i in range(0, n, chunk_size):
                    sl = slice(i, min(i + chunk_size, n))
                    P = shp_points(pts[sl])                # -> 多个 Point 的 GeometryArray
                    d = np.asarray(shp_distance(P, self.multi_poly), dtype=float)
                    if signed:
                        inside = np.asarray(shp_covers(self.multi_poly, P), dtype=bool)
                        d = np.where(inside, -d, d)
                    out[sl] = d
                return out
            except Exception:
                pass  # 进入 fallback

        # --- fallback: 粗略 AABB 近似（没装shapely时） ---
        if not hasattr(self, "aabbs") or len(getattr(self, "aabbs", [])) == 0:
            return np.full((pts.shape[0],), np.inf, dtype=float)

        # AABB 盒内→负号；盒外→到盒边距离（非严格 SDF）
        ds = np.full((pts.shape[0],), np.inf, dtype=float)
        for (xmn, ymn, xmx, ymx) in self.aabbs:
            dx = np.maximum(np.maximum(xmn - pts[:, 0], 0.0), pts[:, 0] - xmx)
            dy = np.maximum(np.maximum(ymn - pts[:, 1], 0.0), pts[:, 1] - ymx)
            d  = np.hypot(dx, dy)
            inside = (pts[:, 0] >= xmn) & (pts[:, 0] <= xmx) & (pts[:, 1] >= ymn) & (pts[:, 1] <= ymx)
            d = np.where(inside, -np.minimum.reduce([pts[:, 0]-xmn, xmx-pts[:, 0], pts[:, 1]-ymn, ymx-pts[:, 1]]), d)
            ds = np.minimum(ds, d)
        return ds


# =============================================================================
# 2-DOF planar arm kinematics & sampling
# =============================================================================
def fk_planar_2link(q, L1=1.0, L2=1.0):
    """Forward kinematics: joint positions and end-effector.
       q: (2,) angles [th1, th2]
       returns: p0(0,0), p1, p2 (end effector)
    """
    th1, th2 = float(q[0]), float(q[1])
    p0 = np.array([0.0, 0.0], dtype=float)
    p1 = np.array([L1*np.cos(th1), L1*np.sin(th1)], dtype=float)
    p2 = p1 + np.array([L2*np.cos(th1+th2), L2*np.sin(th1+th2)], dtype=float)
    return p0, p1, p2

def jacobian_ee(q, L1=1.0, L2=1.0):
    """End-effector Jacobian J(q) ∈ R^{2x2}."""
    th1, th2 = float(q[0]), float(q[1])
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1+th2), np.cos(th1+th2)
    J = np.array([
        [-L1*s1 - L2*s12, -L2*s12],
        [ L1*c1 + L2*c12,  L2*c12]
    ], dtype=float)
    return J

def ik_planar_2link(xy, L1=1.0, L2=1.0, q_seed=None):
    """Analytic IK for 2-link arm. Returns the solution (two branches exist)
       that is closest to q_seed (if given), else elbow-down by default.
       If target out of reach, project to circle of radius L1+L2-eps and solve.
    """
    x, y = float(xy[0]), float(xy[1])
    R = np.hypot(x, y)
    R_max = L1 + L2 - 1e-6
    if R > L1+L2:
        # project back to reachable circle
        x *= R_max / R
        y *= R_max / R
    # law of cosines
    cos2 = (x*x + y*y - L1*L1 - L2*L2)/(2.0*L1*L2)
    cos2 = np.clip(cos2, -1.0, 1.0)
    th2_1 = np.arccos(cos2)
    th2_2 = -th2_1
    k1_1 = L1 + L2*np.cos(th2_1)
    k2_1 = L2*np.sin(th2_1)
    th1_1 = np.arctan2(y, x) - np.arctan2(k2_1, k1_1)
    k1_2 = L1 + L2*np.cos(th2_2)
    k2_2 = L2*np.sin(th2_2)
    th1_2 = np.arctan2(y, x) - np.arctan2(k2_2, k1_2)
    cand = [np.array([th1_1, th2_1]), np.array([th1_2, th2_2])]
    if q_seed is None:
        return cand[0]
    # pick closest to seed (wrap to -pi..pi)
    def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
    diffs = [np.linalg.norm(wrap(c - q_seed)) for c in cand]
    return cand[int(np.argmin(diffs))]

def sample_arm_points_single(q, n1=80, n2=80, L1=1.0, L2=1.0):
    """Discretize the two links into points (small balls centers).
       Returns array (n1+n2, 2)
    """
    p0, p1, p2 = fk_planar_2link(q, L1, L2)
    # link1: from p0 to p1
    s1 = (np.arange(n1, dtype=float)+0.5)/n1  # avoid exactly at joint
    pts1 = p0[None, :] + s1[:, None]*(p1 - p0)[None, :]
    # link2: from p1 to p2
    s2 = (np.arange(n2, dtype=float)+0.5)/n2
    pts2 = p1[None, :] + s2[:, None]*(p2 - p1)[None, :]
    return np.vstack([pts1, pts2])

def ee_from_traj(q_traj, L1=1.0, L2=1.0):
    """Map a trajectory of joints (K,2) to end-effector positions (K,2)."""
    K = q_traj.shape[0]
    out = np.zeros((K, 2), dtype=float)
    for i in range(K):
        _, _, p2 = fk_planar_2link(q_traj[i], L1, L2)
        out[i] = p2
    return out


# =============================================================================
# Online MPC wrapper with Task-Space collision via spheres
# (keeps your state machine & mode colors)
# =============================================================================
class OnlineVPSTOMPCArm:
    def __init__(self,
                 x_goal: np.ndarray,         # task-space goal (2,)
                 q_min: np.ndarray,          # joint limits
                 q_max: np.ndarray,
                 env: TaskSpaceEnvironment,   # polygons in task-space
                 *,
                 ndof: int = 2,
                 N_eval: int = 80,
                 N_via: int = 5,
                 pop_size_warm: int = 48,
                 pop_size_explore: int = 96,
                 sigma_warm: float = 0.08,
                 sigma_explore: float = 4.35,
                 iters_per_tick_warm: int = 6,
                 iters_per_tick_explore: int = 8,
                 vel_lim: np.ndarray = None,
                 acc_lim: np.ndarray = None,
                 dt: float = 0.05,
                 hard_limits: float = 1e6,
                 w_goal: float = 1e3,         # (joint) terminal goal weight (since we IK to exact x_goal)
                 w_smooth: float = 1e-3,
                 # arm & collision sampling
                 L1: float = 1.0,
                 L2: float = 1.0,
                 n_spheres_link1: int = 80,
                 n_spheres_link2: int = 80,
                 sphere_radius: float = 0.03,
                 sdf_margin: float = 0.0,
                 coll_stride: int = 3,       # evaluate collision every N time steps for speed
                 # state machine
                 T_stop: float = 0.8,
                 idle_pos_tol_task: float = 3e-2,
                 idle_vel_tol_task: float = 3e-2,
                 terminal_pos_tol: float = 1e-3,
                 terminal_vel_tol: float = 2e-2):
        self.ndof = ndof
        self.N_eval = N_eval
        self.N_via = N_via
        self.dt = dt
        self.env = env
        self.q_min = np.array(q_min, float)
        self.q_max = np.array(q_max, float)

        self.vel_lim = np.array(vel_lim) if vel_lim is not None else 0.8*np.ones(ndof)
        self.acc_lim = np.array(acc_lim) if acc_lim is not None else 2.0*np.ones(ndof)

        # weights
        self.hard_limits = float(hard_limits)
        self.w_goal = float(w_goal)
        self.w_smooth = float(w_smooth)

        # VPSTO knobs
        self.pop_warm = pop_size_warm
        self.pop_explore = pop_size_explore
        self.sig_warm = sigma_warm
        self.sig_explore = sigma_explore
        self.iters_warm = iters_per_tick_warm
        self.iters_explore = iters_per_tick_explore

        # arm & collision parameters
        self.L1 = float(L1); self.L2 = float(L2)
        self.n1 = int(n_spheres_link1); self.n2 = int(n_spheres_link2)
        self.sphere_r = float(sphere_radius)
        self.sdf_margin = float(sdf_margin)
        self.coll_stride = max(1, int(coll_stride))
        self.time_idx = np.arange(0, self.N_eval, self.coll_stride, dtype=int)

        # goal in task space & IK → joint goal (fixed branch near initial q)
        self.x_goal = np.asarray(x_goal, float).reshape(2)

        # state machine parameters
        self.T_stop = float(T_stop)

        # stateful: previous best solution for warm-start / idle
        self.prev_sol = None
        self.prev_T = None

        # VP trajectory tools
        self.vptraj = VPTraj(ndof, N_eval, N_via, self.vel_lim, self.acc_lim)
        self.vptraj_direct = VPTraj(ndof, N_eval, 1, self.vel_lim, self.acc_lim)

        # logs (for animation)
        self.trajectory_log = []      # store planned EE path (K,2)
        self.candidate_log = []       # store raw q candidates (dict) → we convert to EE in animate
        self.best_via_log = []        # via in joint space (could skip in EE plot)
        self.mode_log = []
        self.compute_time_log = []

        # modes / thresholds in TASK space
        self.idle = False
        self._idle_cnt = 0
        self.idle_pos_tol = float(idle_pos_tol_task)
        self.idle_vel_tol = float(idle_vel_tol_task)
        self.idle_enter_count = 2
        self.idle_exit_pos = 2.5*self.idle_pos_tol
        self.idle_exit_vel = 3.0*self.idle_vel_tol

        self._terminal_hold = False
        self.terminal_pos_tol = float(terminal_pos_tol)
        self.terminal_vel_tol = float(terminal_vel_tol)
        self.terminal_release_pos = 3*self.terminal_pos_tol
        self.terminal_release_vel = self.idle_exit_vel

        self._zero_latch = False
        self._direct_traj_q = None      # (K,2)
        self._direct_traj_ee = None     # (K,2)
        self._direct_traj_dq = None     # (K,2)
        self._direct_idx = 0
        self._direct_stride = 1
        self._direct_T_rem = 0.0

        # initialize q_goal (joint) from IK using a reasonable seed (will be set in first step)
        self.q_goal = None

    # ---------- helpers ----------
    def _enter_idle(self): self.idle = True; self._idle_cnt = 0
    def _exit_idle(self):  self.idle = False; self._idle_cnt = 0

    def _bounds_violation(self, q):  # q: (B,N,2)
        below = (q < self.q_min[None, None, :]).sum(axis=(1, 2))
        above = (q > self.q_max[None, None, :]).sum(axis=(1, 2))
        return below + above

    def _smoothness(self, dq, ddq):   # (B,N,2)
        v2 = (dq**2).sum(-1)
        a2 = (ddq**2).sum(-1)
        va = (dq*ddq).sum(-1)
        kappa2 = (v2*a2 - va**2) / (np.maximum(v2, 1e-8)**3)
        return kappa2.mean(axis=1)

    def _collision_cost_batch(self, q_batch: np.ndarray) -> np.ndarray:
        """
        q_batch: (B, K, 2) batch of joint angles
        return: (B,) collision penalty (vectorized)
        """
        B, K, _ = q_batch.shape
        # 选取时间子采样索引，避免每个tick评估全部 N_eval
        if not hasattr(self, "time_idx") or self.time_idx is None:
            stride = getattr(self, "coll_stride", 5)
            self.time_idx = np.arange(0, K, max(1, int(stride)), dtype=int)
        tidx = self.time_idx
        Ksub = tidx.size

        # 小球参数
        S1 = getattr(self, "spheres_per_link", 40)
        S2 = S1
        r  = float(getattr(self, "sphere_r", 0.03))
        # 均匀布点 [0,1]（不把末端算两次）
        u1 = np.linspace(0.0, 1.0, S1, endpoint=False, dtype=float)
        u2 = np.linspace(0.0, 1.0, S2, endpoint=False, dtype=float)

        # 取子采样 q
        qsub = q_batch[:, tidx, :]                # (B, Ksub, 2)
        q1   = qsub[..., 0]                       # (B, Ksub)
        q12  = qsub[..., 0] + qsub[..., 1]        # (B, Ksub)

        # 预计算三角函数（广播到 (B,Ksub,1) 便于与 u 扩展）
        c1, s1   = np.cos(q1)[..., None],  np.sin(q1)[..., None]       # (B,Ksub,1)
        c12, s12 = np.cos(q12)[..., None], np.sin(q12)[..., None]

        # 链长（如你在 __init__ 里有 self.L1, self.L2 就用它们）
        L1 = float(getattr(self, "L1", 1.0))
        L2 = float(getattr(self, "L2", 1.0))

        # link1 上 S1 个小球的坐标： base + u*L1*[cos(q1), sin(q1)]
        # 生成形状 (1,1,S1) 的 u，以便广播到 (B,Ksub,S1)
        U1 = u1[None, None, :]
        P1x = (U1 * L1) * c1
        P1y = (U1 * L1) * s1

        # link2 关节位置 (B,Ksub,1,2)
        J2x = (L1 * c1)
        J2y = (L1 * s1)

        # link2 上 S2 个小球的坐标： J2 + u*L2*[cos(q1+q2), sin(q1+q2)]
        U2 = u2[None, None, :]
        P2x = J2x + (U2 * L2) * c12
        P2y = J2y + (U2 * L2) * s12

        # 拼成 (B, Ksub, S1+S2, 2)
        X1 = np.stack([P1x, P1y], axis=-1)             # (B,Ksub,S1,2)
        X2 = np.stack([P2x, P2y], axis=-1)             # (B,Ksub,S2,2)
        X  = np.concatenate([X1, X2], axis=2)          # (B,Ksub,S,2)
        S  = X.shape[2]

        # 展平到 (B*Ksub*S, 2) 一次性调用 env.distance（内部已向量化+分块）
        Xf = X.reshape(-1, 2)
        d  = self.env.distance(Xf, signed=True)        # (B*Ksub*S,)
        d  = d.reshape(B, Ksub, S)

        # 安全余量（小球半径），把在障碍内/很近的点惩罚： hinge(- (d - r))
        phi = d - r
        pen = np.clip(-phi, 0.0, None)                 # (B,Ksub,S)
        # 你可以用 L1, L2 等权重，这里用均值再平方（更平滑）
        cost = (pen.mean(axis=(1, 2)))**2              # (B,)

        return cost


    def _mpc_loss(self, cand):
        q, dq, ddq, T = cand['pos'], cand['vel'], cand['acc'], cand['T']  # (B,N,2), (B,)
        # terminal joint error (since q_goal from IK to hit x_goal exactly)
        term = np.linalg.norm(q[:, -1, :] - self.q_goal[None, :], axis=1)
        # limits
        bounds = self._bounds_violation(q)
        # smoothness
        smooth = self._smoothness(dq, ddq)
        # collision in task space
        coll = self._collision_cost_batch(q)
        return (T
                + self.w_smooth * smooth
                + self.hard_limits * bounds
                + self.w_goal * term
                + 1.0 * coll)   # collision weight already squared; keep factor 1.0 (tuneable)

    # ---------- warm-start p vector (qT given case) ----------
    def _time_shifted_p(self, sol, dt):
        T = max(float(getattr(sol, 'T_best', 0.0)), 1e-3)
        if T <= dt: return None
        t_grid = np.linspace(dt, T, self.N_via + 1)   # N_via segments remain
        q_grid, _, _ = sol.get_posvelacc(t_grid)
        return q_grid[1:-1].reshape(-1)  # (ndof*(N_via-1),)

    # ---------- zero-via to current IK target ----------
    def _try_direct(self, q, dq, q_goal_joint, T_stop_local):
        P0 = np.zeros((1, 0))
        dqT = np.zeros(2)
        T0 = float(self.vptraj_direct.get_min_duration(P0, q0=q, dq0=dq, qT=q_goal_joint, dqT=dqT))
        if T0 <= 0 or T0 > T_stop_local:
            return False, None, None, None, None, None, None
        q_traj, dq_traj, _ = self.vptraj_direct.get_trajectory(P0, q0=q, dq0=dq, qT=q_goal_joint, dqT=dqT, T=T0)
        q_traj = q_traj[0]; dq_traj = dq_traj[0]
        # quick collision screen on subsampled times
        feas = True
        for k in self.time_idx:
            pts = sample_arm_points_single(q_traj[k], self.n1, self.n2, self.L1, self.L2)
            d = self.env.distance(pts)
            if np.any(d < (self.sphere_r - 1e-6)):  # any penetration
                feas = False; break
        if not feas:
            return False, None, None, None, None, None, None
        # sample Δt
        stride = max(1, int(round(self.dt / T0 * (self.N_eval - 1))))
        k = stride
        qdt = q_traj[k].copy()
        dqdt = dq_traj[k].copy()
        return True, qdt, dqdt, q_traj, dq_traj, T0, stride

    def _roll_zero_latch(self, tic):
        if self._direct_traj_q is None or self._direct_traj_ee is None:
            self._zero_latch = False
            return None
        N = self._direct_traj_q.shape[0]
        if (self._direct_idx >= N-1) or (self._direct_T_rem <= 0.0):
            # end: if very close in task space, lock terminal
            dist_tail = np.linalg.norm(self._direct_traj_ee[-1] - self.x_goal)
            if dist_tail <= self.terminal_pos_tol:
                self._terminal_hold = True
                qdt = self.q_goal.copy() if self.q_goal is not None else self._direct_traj_q[-1].copy()
                dqdt = np.zeros_like(qdt)
                self.trajectory_log.append(self._direct_traj_ee)
                self.candidate_log.append(None)
                self.best_via_log.append(None)
                self.mode_log.append('idle')
                self.compute_time_log.append(time.perf_counter()-tic)
                self._zero_latch = False
                return qdt, dqdt
            self._zero_latch = False
            return None
        # advance one Δt
        nxt = min(self._direct_idx + self._direct_stride, N-1)
        qdt = self._direct_traj_q[nxt].copy()
        dqdt = self._direct_traj_dq[nxt].copy()
        self._direct_idx = nxt
        self._direct_T_rem -= self.dt
        # log EE path (full)
        self.trajectory_log.append(self._direct_traj_ee)
        self.candidate_log.append(None)
        self.best_via_log.append(None)
        # mode: if close → idle, else zero (task-space)
        ee = self._direct_traj_ee[nxt]
        J = jacobian_ee(qdt, self.L1, self.L2)
        vee = np.linalg.norm(J @ dqdt)
        if (np.linalg.norm(ee - self.x_goal) < self.idle_pos_tol) and (vee < self.idle_vel_tol):
            self.mode_log.append('idle')
            self._terminal_hold = True
        else:
            self.mode_log.append('zero')
        self.compute_time_log.append(time.perf_counter()-tic)
        return qdt, dqdt

    # ---------- one MPC tick ----------
    def step(self, q: np.ndarray, dq: np.ndarray):
        tic = time.perf_counter()
        q = np.asarray(q, float).reshape(2)
        dq = np.asarray(dq, float).reshape(2)

        # define/update q_goal by IK near current q (keep branch)
        if self.q_goal is None:
            self.q_goal = ik_planar_2link(self.x_goal, self.L1, self.L2, q_seed=q)
        else:
            # slowly re-choose nearest branch to current q (robust if branch flips)
            self.q_goal = ik_planar_2link(self.x_goal, self.L1, self.L2, q_seed=q)

        # Terminal guard in task space
        p0, p1, ee = fk_planar_2link(q, self.L1, self.L2)
        J = jacobian_ee(q, self.L1, self.L2)
        dist_now = np.linalg.norm(ee - self.x_goal)
        vee_now = np.linalg.norm(J @ dq)
        if self._terminal_hold:
            if (dist_now <= self.terminal_release_pos) and (vee_now <= self.terminal_release_vel):
                qdt = self.q_goal.copy()
                dqdt = np.zeros_like(qdt)
                # log short EE segment
                ee_goal = self.x_goal.copy()
                self.trajectory_log.append(np.vstack([ee, ee_goal]))
                self.candidate_log.append(None)
                self.best_via_log.append(None)
                self.mode_log.append('idle')
                self.compute_time_log.append(time.perf_counter()-tic)
                return qdt, dqdt
            else:
                self._terminal_hold = False

        # Zero latch roll
        if self._zero_latch:
            rolled = self._roll_zero_latch(tic)
            if rolled is not None:
                return rolled

        # Idle-tail of VPSTO (task-space conditions)
        mode_warmstart = (self.prev_sol is not None) and (self.prev_T is not None) and (self.prev_T > 2.0*self.dt)
        if self.idle:
            if (dist_now > self.idle_exit_pos) or (vee_now > self.idle_exit_vel) or \
               (self.prev_sol is None) or (self.prev_T is None) or (self.prev_T <= 0.0):
                self._exit_idle()
            else:
                qdt, dqdt, _ = self.prev_sol.get_posvelacc(self.dt)
                qdt = qdt[0]; dqdt = dqdt[0]
                self.prev_T -= self.dt
                # log planned EE
                t_full = np.linspace(0.0, max(self.prev_T, self.dt), self.N_eval)
                q_traj, _, _ = self.prev_sol.get_posvelacc(t_full)
                self.trajectory_log.append(ee_from_traj(q_traj, self.L1, self.L2))
                self.candidate_log.append(None)
                self.best_via_log.append(None)
                # near terminal → lock
                ee_next = fk_planar_2link(qdt, self.L1, self.L2)[2]
                vee_next = np.linalg.norm(jacobian_ee(qdt, self.L1, self.L2) @ dqdt)
                if (np.linalg.norm(ee_next - self.x_goal) < self.terminal_pos_tol) and (vee_next < self.terminal_vel_tol):
                    self._terminal_hold = True
                self.mode_log.append('idle')
                self.compute_time_log.append(time.perf_counter()-tic)
                return qdt, dqdt

        # Zero-try to IK goal (with time lower bound)
        vmax = float(np.linalg.norm(self.vel_lim)) + 1e-9
        T_stop_local = max(2.0*self.dt, min( self.T_stop, 2.5 * dist_now / vmax ))
        ok, qdt_d, dqdt_d, qtraj_d, dqtraj_d, T0, stride = self._try_direct(q, dq, self.q_goal, T_stop_local)
        if ok:
            self._zero_latch = True
            self._direct_traj_q = qtraj_d
            self._direct_traj_dq = dqtraj_d
            self._direct_traj_ee = ee_from_traj(qtraj_d, self.L1, self.L2)
            self._direct_stride   = max(1, int(stride))
            self._direct_idx      = self._direct_stride
            self._direct_T_rem    = max(0.0, T0 - self.dt)
            # log planned EE (full)
            self.trajectory_log.append(self._direct_traj_ee)
            self.candidate_log.append(None)
            self.best_via_log.append(None)
            # if already near goal, lock terminal
            ee_next = fk_planar_2link(qdt_d, self.L1, self.L2)[2]
            vee_next = np.linalg.norm(jacobian_ee(qdt_d, self.L1, self.L2) @ dqdt_d)
            if (np.linalg.norm(ee_next - self.x_goal) < self.idle_pos_tol) and (vee_next < self.idle_vel_tol):
                self._terminal_hold = True
                self.mode_log.append('idle')
            else:
                self.mode_log.append('zero')
            self.compute_time_log.append(time.perf_counter()-tic)
            return qdt_d, dqdt_d

        # VP-STO full optimization (qT and dqT GIVEN; qT is IK to hit x_goal)
        opt = VPSTOOptions(self.ndof)
        opt.vel_lim = self.vel_lim.copy()
        opt.acc_lim = self.acc_lim.copy()
        opt.N_eval = self.N_eval
        opt.N_via = self.N_via
        opt.CMA_diagonal = True
        opt.pop_size = self.pop_warm if mode_warmstart else self.pop_explore
        opt.sigma_init = self.sig_warm if mode_warmstart else self.sig_explore
        opt.max_iter = self.iters_warm if mode_warmstart else self.iters_explore
        opt.log = False; opt.verbose = False

        solver = VPSTO(opt)
        if mode_warmstart:
            p_init = self._time_shifted_p(self.prev_sol, self.dt)
            if p_init is not None and p_init.size == self.ndof*(self.N_via-1):
                solver.set_initial_guess(p_init)

        dqT = np.zeros(self.ndof, dtype=float)
        sol = solver.minimize(self._mpc_loss, q0=q, dq0=dq, qT=self.q_goal, dqT=dqT, T=None)
        Tbest = float(getattr(sol, 'T_best', 0.0))

        # candidates (we will convert to EE at draw time for speed)
        if hasattr(sol, 'candidates') and isinstance(sol.candidates, dict) and sol.candidates.get('pos', None) is not None:
            try:
                self.candidate_log.append({
                    'pos': sol.candidates['pos'].copy(),  # (pop,N,2)
                    'vel': sol.candidates['vel'].copy(),
                    'T':   sol.candidates['T'].copy()
                })
            except Exception:
                self.candidate_log.append(None)
        else:
            self.candidate_log.append(None)

        # via points (joint space)
        via_best = None
        try:
            if hasattr(sol, 'p_best') and sol.p_best is not None and self.N_via > 1:
                via_best = sol.p_best.reshape(self.N_via-1, self.ndof).copy()
        except Exception:
            via_best = None

        # handle degenerate T
        if Tbest <= 0:
            self.trajectory_log.append(None)
            self.mode_log.append('warm' if mode_warmstart else 'exploration')
            self.best_via_log.append(via_best)
            self.compute_time_log.append(time.perf_counter()-tic)
            return q.copy(), np.zeros_like(dq)

        # extract Δt
        qdt, dqdt, _ = sol.get_posvelacc(self.dt)
        qdt = qdt[0]; dqdt = dqdt[0]

        # log full planned EE path
        t_full = np.linspace(0, Tbest, self.N_eval)
        q_traj, _, _ = sol.get_posvelacc(t_full)
        self.trajectory_log.append(ee_from_traj(q_traj, self.L1, self.L2))

        # update warm-start state
        self.prev_sol = sol
        self.prev_T = Tbest - self.dt

        # idle enter logic in task space
        ee_next = fk_planar_2link(qdt, self.L1, self.L2)[2]
        vee_next = np.linalg.norm(jacobian_ee(qdt, self.L1, self.L2) @ dqdt)
        if (np.linalg.norm(ee_next - self.x_goal) < self.idle_pos_tol) and (vee_next < self.idle_vel_tol):
            self._idle_cnt += 1
        else:
            self._idle_cnt = 0
        if (not self.idle) and (self._idle_cnt >= 2):
            self._enter_idle()
        # extremely close → lock terminal for next tick
        if (np.linalg.norm(ee_next - self.x_goal) < self.terminal_pos_tol) and (vee_next < self.terminal_vel_tol):
            self._terminal_hold = True

        self.mode_log.append('warm' if mode_warmstart else 'exploration')
        self.best_via_log.append(via_best)
        self.compute_time_log.append(time.perf_counter()-tic)
        return qdt, dqdt


# =============================================================================
# Demo
# =============================================================================
def run_demo():
    # ---- Task-space obstacles in workspace (units: meters)
    # workspace roughly within radius 2.0 (since L1=L2=1). Put a few polygons.
    polys = [
        np.array([[-0.2,  0.8], [0.2,  0.9], [0.15, 1.2], [-0.25, 1.15]]),     # near top-left
        np.array([[ 0.6,  0.2], [1.0,  0.2], [1.0, 0.6], [0.6,  0.6]]),        # square on right
        np.array([[-0.9, -0.2], [-0.5, -0.3], [-0.6, 0.2]])                    # triangle
    ]
    env = TaskSpaceEnvironment(polys)

    # ---- Joint limits and dynamics limits
    q_min = np.array([-np.pi, -np.pi])
    q_max = np.array([ np.pi,  np.pi])

    # ---- Start and goal (task space)
    q0 = np.array([0.1, 0.2])  # initial joints
    dq0 = np.zeros(2)
    _, _, x0 = fk_planar_2link(q0)
    x_goal = np.array([0.5, -0.5])  # reachable target

    ctrl = OnlineVPSTOMPCArm(
        x_goal=x_goal,
        q_min=q_min, q_max=q_max,
        env=env,
        ndof=2, N_eval=80, N_via=3,
        pop_size_warm=48, pop_size_explore=96,
        sigma_warm=0.08, sigma_explore=4.35,
        iters_per_tick_warm=6, iters_per_tick_explore=8,
        vel_lim=np.array([1.0, 1.0]),
        acc_lim=np.array([4.0, 4.0]),
        dt=0.05,
        hard_limits=1e8,
        w_goal=1e3, w_smooth=1e-3,
        L1=1.0, L2=1.0,
        n_spheres_link1=100,
        n_spheres_link2=100,
        sphere_radius=0.035,
        sdf_margin=0.0,
        coll_stride=3,
        T_stop=0.8,
        idle_pos_tol_task=3e-2,
        idle_vel_tol_task=3e-2,
        terminal_pos_tol=1e-3,
        terminal_vel_tol=2e-2
    )

    # ---- Perfect tracking simulation
    Tsim = 3.0
    steps = int(Tsim/ctrl.dt)
    q_hist = [q0.copy()]
    dq_hist = [dq0.copy()]

    t0 = time.perf_counter()
    for _ in range(steps):
        q_ref, dq_ref = ctrl.step(q_hist[-1], dq_hist[-1])
        q_hist.append(q_ref.copy()); dq_hist.append(dq_ref.copy())
        # early stop (task-space)
        _, _, ee = fk_planar_2link(q_hist[-1])
        J = jacobian_ee(q_hist[-1], ctrl.L1, ctrl.L2)
        vee = np.linalg.norm(J @ dq_hist[-1])
        if np.linalg.norm(ee - x_goal) < 1e-2 and vee < 1e-2:
            break
    t1 = time.perf_counter()

    q_hist = np.array(q_hist); dq_hist = np.array(dq_hist)
    t = np.arange(len(q_hist))*ctrl.dt

    # ---- quick plots
    print(f"Sim time {t1-t0:.2f}s, {len(q_hist)} steps.")
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    # Workspace plot
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlim(-2.1, 2.1); ax0.set_ylim(-2.1, 2.1)
    ax0.grid(True, alpha=0.3)
    env.draw(ax0)
    # start arm pose
    p0, p1, p2 = fk_planar_2link(q_hist[0])
    ax0.plot([p0[0], p1[0], p2[0]],[p0[1], p1[1], p2[1]], 'k-o', lw=2, ms=4)
    # goal
    ax0.scatter(x_goal[0], x_goal[1], s=120, c='red', marker='*', label='Goal')
    ax0.legend(loc='lower right')
    ax0.set_title('Workspace (arm)')

    # Velocity (joint)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, dq_hist[:, 0], label='dq1')
    ax1.plot(t, dq_hist[:, 1], label='dq2')
    ax1.axhline(+ctrl.vel_lim[0], ls='--', c='r', lw=1)
    ax1.axhline(-ctrl.vel_lim[0], ls='--', c='r', lw=1)
    ax1.grid(True, alpha=0.3); ax1.legend(); ax1.set_ylabel('joint vel')

    # Task-space distance to goal
    ax2 = fig.add_subplot(gs[1, 1])
    d_task = []
    for qi, dqi in zip(q_hist, dq_hist):
        _, _, ee = fk_planar_2link(qi)
        d_task.append(np.linalg.norm(ee - x_goal))
    d_task = np.array(d_task)
    ax2.plot(t, d_task); ax2.set_xlabel('time (s)'); ax2.set_ylabel('‖ee - x_goal‖')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()

    print(f"Finished in {t[-1]:.2f}s, final task error {d_task[-1]:.4f}")

    # ---- animation (with optional saving)
    print("Creating animation...")
    save_path = None
    if SAVE_ANIMATION:
        save_path = ANIMATION_FILENAME
        print(f"Animation will be saved to: {save_path}")
    else:
        try:
            ans = input("Save animation? (y/n): ").strip().lower()
            if ans in ['y','yes']:
                fn = input("Filename (.mp4 or .gif, default mpc_arm_animation.mp4): ").strip()
                if not fn: fn = "mpc_arm_animation.mp4"
                if not (fn.lower().endswith('.mp4') or fn.lower().endswith('.gif')):
                    fn += '.mp4'
                save_path = fn
        except (EOFError, KeyboardInterrupt):
            print("Non-interactive mode: not saving.")
    create_animation(q_hist, dq_hist, ctrl, env, t, q0, x_goal, save_path)


def create_animation(q_hist, dq_hist, ctrl, env, time_vec, q0, x_goal, save_path=None):
    MODE_COLORS = {'zero':'tab:green','idle':'tab:gray','warm':'tab:orange','exploration':'tab:purple',None:'tab:purple'}

    fig, (axw, axv) = plt.subplots(1, 2, figsize=(17, 8))

    # Workspace panel
    axw.set_facecolor('white')
    axw.set_xlim(-2.1, 2.1); axw.set_ylim(-2.1, 2.1)
    axw.set_aspect('equal', adjustable='box'); axw.grid(True, alpha=0.3)
    axw.set_xlabel('X'); axw.set_ylabel('Y'); axw.set_title('2-DOF Arm MPC in Workspace')

    env.draw(axw)
    axw.scatter(x_goal[0], x_goal[1], c='red', s=120, marker='*', label='Goal')
    axw.legend(loc='upper right')

    # Arm artist (links & joints)
    link_line, = axw.plot([], [], 'k-o', lw=3, ms=5, alpha=0.9, label='arm')

    # Executed EE path
    ee_exec_line, = axw.plot([], [], color='tab:blue', lw=2.5, alpha=0.9, label='EE executed')

    # Planned EE path (mode-colored)
    ee_plan_line, = axw.plot([], [], color='tab:purple', lw=2.0, alpha=0.9, label='EE planned')

    # Candidate EE paths
    candidate_lines = []
    max_cand = 40
    for _ in range(max_cand):
        ln, = axw.plot([], [], color='orange', lw=0.8, alpha=0.25)
        candidate_lines.append(ln)

    # Status text
    time_text = axw.text(0.02, 0.98, '', transform=axw.transAxes, fontsize=12,
                         va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    status_text = axw.text(0.02, 0.80, '', transform=axw.transAxes, fontsize=10,
                           va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Velocity panel (joint)
    axv.set_title('Joint velocities'); axv.grid(True, alpha=0.3)
    axv.set_xlabel('Time (s)'); axv.set_ylabel('Velocity')
    axv.axhline(+ctrl.vel_lim[0], ls='--', c='r', lw=1, alpha=0.7)
    axv.axhline(-ctrl.vel_lim[0], ls='--', c='r', lw=1, alpha=0.7)
    dq1_line, = axv.plot([], [], 'b-', lw=2, label='dq1')
    dq2_line, = axv.plot([], [], 'r-', lw=2, label='dq2')
    cur_time_line = axv.axvline(0, color='k', lw=1.0, alpha=0.7)
    axv.legend()

    # cache executed EE path
    ee_exec = []
    for q in q_hist:
        ee_exec.append( fk_planar_2link(q, ctrl.L1, ctrl.L2)[2] )
    ee_exec = np.array(ee_exec)

    def init():
        link_line.set_data([], [])
        ee_exec_line.set_data([], [])
        ee_plan_line.set_data([], [])
        for ln in candidate_lines:
            ln.set_data([], [])
        dq1_line.set_data([], []); dq2_line.set_data([], [])
        time_text.set_text(''); status_text.set_text('')
        return [link_line, ee_exec_line, ee_plan_line, *candidate_lines,
                dq1_line, dq2_line, time_text, status_text, cur_time_line]

    def animate(frame):
        k = min(frame, len(q_hist)-1)
        tcur = time_vec[k]

        # arm pose
        p0, p1, p2 = fk_planar_2link(q_hist[k], ctrl.L1, ctrl.L2)
        link_line.set_data([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])

        # executed EE path so far
        if k > 0:
            ee_exec_line.set_data(ee_exec[:k+1, 0], ee_exec[:k+1, 1])

        # planned EE path (from controller logs)
        if k < len(ctrl.trajectory_log) and ctrl.trajectory_log[k] is not None:
            ee_plan = ctrl.trajectory_log[k]
            ee_plan_line.set_data(ee_plan[:, 0], ee_plan[:, 1])
        else:
            ee_plan_line.set_data([], [])
        # color by mode
        mode = ctrl.mode_log[k] if k < len(ctrl.mode_log) else None
        ee_plan_line.set_color(MODE_COLORS.get(mode, MODE_COLORS[None]))

        # candidate EE paths (convert on-the-fly)
        for ln in candidate_lines:
            ln.set_data([], [])
        if k < len(ctrl.candidate_log) and ctrl.candidate_log[k] is not None:
            cand = ctrl.candidate_log[k]
            Q = cand['pos']  # (pop,N,2)
            n_show = min(Q.shape[0], len(candidate_lines), 30)
            for i in range(n_show):
                ee_c = ee_from_traj(Q[i], ctrl.L1, ctrl.L2)
                candidate_lines[i].set_data(ee_c[:, 0], ee_c[:, 1])

        # velocity panel
        if k > 0:
            dq1_line.set_data(time_vec[:k+1], dq_hist[:k+1, 0])
            dq2_line.set_data(time_vec[:k+1], dq_hist[:k+1, 1])
            axv.set_xlim(0, max(time_vec[k]+1, 2))
            vmax = max(np.abs(dq_hist[:k+1]).max(), ctrl.vel_lim[0])*1.1
            axv.set_ylim(-vmax, vmax)
        cur_time_line.set_xdata([tcur, tcur])

        # overlays
        speed_task = np.linalg.norm( jacobian_ee(q_hist[k], ctrl.L1, ctrl.L2) @ dq_hist[k] )
        dist_task = np.linalg.norm( fk_planar_2link(q_hist[k], ctrl.L1, ctrl.L2)[2] - x_goal )
        # compute-time stats
        if k < len(ctrl.compute_time_log):
            cur_ms = ctrl.compute_time_log[k]*1000.0
            cur_hz = (1.0 / ctrl.compute_time_log[k]) if ctrl.compute_time_log[k] > 1e-9 else float('inf')
            avg_ms = np.mean(ctrl.compute_time_log[:k+1])*1000.0
            avg_hz = 1.0 / (avg_ms/1000.0) if avg_ms>1e-9 else float('inf')
        else:
            cur_ms = avg_ms = 0.0; cur_hz = avg_hz = 0.0
        total = k+1
        def pct(lbl): 
            return int(round(100.0*ctrl.mode_log[:k+1].count(lbl)/total)) if total>0 else 0
        zf, wf, ef, ifq = pct('zero'), pct('warm'), pct('exploration'), pct('idle')

        time_text.set_text(f"Time: {tcur:.2f}s\nStep: {k}/{len(q_hist)-1}")
        status_text.set_text(
            f"Mode: {mode}\n"
            f"Speed(task): {speed_task:.3f}\nDist→Goal: {dist_task:.4f}\n"
            f"Compute: cur {cur_ms:.1f} ms ({cur_hz:.1f} Hz) | avg {avg_ms:.1f} ms ({avg_hz:.1f} Hz)\n"
            f"Freq: Z {zf}% | W {wf}% | E {ef}% | I {ifq}%"
        )

        return [link_line, ee_exec_line, ee_plan_line, *candidate_lines,
                dq1_line, dq2_line, time_text, status_text, cur_time_line]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(q_hist), interval=50, blit=False, repeat=True)

    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        try:
            if save_path.lower().endswith('.gif'):
                Writer = animation.writers['pillow']; writer = Writer(fps=15)
                anim.save(save_path, writer=writer)
            else:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=20, metadata=dict(artist='VP-STO MPC'), bitrate=2000)
                anim.save(save_path, writer=writer)
            print("Animation saved.")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            if save_path.lower().endswith('.mp4'):
                print("Install ffmpeg: sudo apt-get install ffmpeg")
            else:
                print("Install pillow: pip install pillow")

    plt.tight_layout(); plt.show()
    return anim


if __name__ == "__main__":
    run_demo()
