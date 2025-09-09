import os
import sys
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

# --- optional project path (keeps compatibility with your repo layout) ------
try:
    from config import PLANNER_PATH
    if PLANNER_PATH not in sys.path:
        sys.path.append(PLANNER_PATH)
except Exception:
    pass

# --- VP‑STO core -------------------------------------------------------------
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
        # 左下区域
        self.poly_list.append(np.array([
            [0.10, 0.13], [0.23, 0.12], [0.19, 0.28], [0.10, 0.32], [0.16, 0.20]
        ]))
        # 中上区域
        self.poly_list.append(np.array([
            [0.25, 0.34], [0.31, 0.35], [0.32, 0.41], [0.27, 0.44], [0.23, 0.40]
        ]))
        # 右下区域
        self.poly_list.append(np.array([
            [0.35, 0.12], [0.38, 0.10], [0.41, 0.11], [0.42, 0.21], [0.35, 0.24]
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
# Online MPC wrapper around VP‑STO
#   • Full‑horizon optimization each control tick via VPSTO.minimize
#   • "previous_sol" warm‑start by time‑shifting the prior best plan
#   • "predictive_sampling" = larger sigma when no warm start is available
#   • emits a short reference (Δt) and shifts the remaining plan
# =============================================================================
class OnlineVPSTOMPC:
    def __init__(self,
                 q_goal: np.ndarray,
                 q_min: np.ndarray,
                 q_max: np.ndarray,
                 env: CollisionEnvironment,
                 *,
                 ndof: int = 2,
                 N_eval: int = 80,
                 N_via: int = 6,
                 pop_size_warm: int = 64,
                 pop_size_explore: int = 96,
                 sigma_warm: float = 0.08,
                 sigma_explore: float = 0.35,
                 iters_per_tick_warm: int = 6,
                 iters_per_tick_explore: int = 8,
                 vel_lim: np.ndarray = None,
                 acc_lim: np.ndarray = None,
                 dt: float = 0.05,
                 hard_collision: float = 1e6,
                 hard_limits: float = 1e6,
                 soft_goal: float = 1e3,
                 smooth_w: float = 1e-3):
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

        # stateful: previous (best) plan for warm start
        self.prev_sol = None   # a VPSTOSolution
        self.prev_T = None

        # trajectory utilities
        self.vptraj = VPTraj(ndof, N_eval, N_via, self.vel_lim, self.acc_lim)

        # logging for animation
        self.trajectory_log = []  # Log of planned trajectories
        self.candidate_log = []   # Log of candidate trajectories

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
        # cand has keys: 'pos', 'vel', 'acc', 'T' as in VPSTO.minimize
        q, dq, ddq, T = cand['pos'], cand['vel'], cand['acc'], cand['T']
        goal_err = np.linalg.norm(q[:, -1, :] - self.q_goal[None, :], axis=1)
        bounds_v = self._bounds_violation(q)
        coll = self._collision_cost(q)
        smooth = self._smoothness(dq, ddq)
        return (T
                + self.smooth_w * smooth
                + self.hard_limits * bounds_v
                + self.hard_collision * coll
                + self.soft_goal * goal_err)

    # ---------- warm start: shift previous plan by Δt -----------------------
    def _time_shifted_p(self, sol, dt: float):
        """Build p_init for the next tick by sampling the previous best plan
        on an evenly spaced time-grid starting at t=dt.

        Assumes we call VPSTO with qT and dqT **given** (so p are the N_via-1
        internal via positions)."""
        T = max(float(getattr(sol, 'T_best', 0.0)), 1e-3)
        if T <= dt:
            return None
        t_grid = np.linspace(dt, T, self.N_via + 1)  # N_via segments on the remainder
        q_grid, _, _ = sol.get_posvelacc(t_grid)
        # internal nodes only → flatten
        p_init = q_grid[1:-1].reshape(-1)
        return p_init

    # ---------- one MPC tick -----------------------------------------------
    def step(self, q: np.ndarray, dq: np.ndarray):
        # --- shape guard: ensure 1-D (ndof,) arrays to avoid 0-D concat inside VPSTO ---
        q = np.asarray(q, dtype=float).reshape(self.ndof)
        dq = np.asarray(dq, dtype=float).reshape(self.ndof)
        self.q_goal = np.asarray(self.q_goal, dtype=float).reshape(self.ndof)
        have_warm = self.prev_sol is not None and self.prev_T is not None and self.prev_T > 0.0

        # Configure VP‑STO options per mode
        opt = VPSTOOptions(self.ndof)
        opt.vel_lim = self.vel_lim.copy()
        opt.acc_lim = self.acc_lim.copy()
        opt.N_eval = self.N_eval
        opt.N_via = self.N_via
        opt.CMA_diagonal = True  # faster sep‑CMA‑ES
        opt.pop_size = self.pop_warm if have_warm else self.pop_explore
        opt.sigma_init = self.sig_warm if have_warm else self.sig_explore
        opt.max_iter = self.iters_warm if have_warm else self.iters_explore
        opt.log = False
        opt.verbose = False

        # Build solver and (optionally) warm start
        solver = VPSTO(opt)
        if have_warm:
            p_init = self._time_shifted_p(self.prev_sol, self.dt)
            if p_init is not None and p_init.size == self.ndof*(self.N_via-1):
                solver.set_initial_guess(p_init)

        # Run VP‑STO over FULL horizon; we provide qT=goal, dqT=0 (both **given**)
        dqT = np.zeros(self.ndof, dtype=float)
        sol = solver.minimize(self._mpc_loss, q0=q, dq0=dq, qT=self.q_goal, dqT=dqT, T=None)
        Tbest = float(getattr(sol, 'T_best', 0.0))
        
        # Log candidates for animation (from the last iteration)
        if hasattr(sol, 'candidates') and sol.candidates['pos'] is not None:
            self.candidate_log.append({
                'pos': sol.candidates['pos'].copy(),
                'vel': sol.candidates['vel'].copy(),
                'T': sol.candidates['T'].copy()
            })
        else:
            self.candidate_log.append(None)
        
        if Tbest <= 0:
            # if something went wrong, just hold
            return q.copy(), np.zeros_like(dq)

        if Tbest <= self.dt:
            # horizon collapses → jump to the end
            q_next, dq_next, _ = sol.get_posvelacc(Tbest)
            q_next = q_next[0]
            dq_next = dq_next[0]
            dq_next = np.zeros_like(q_next)
            self.prev_sol = None
            self.prev_T = None
            return q_next, dq_next

        # short‑horizon reference and warm start for next tick
        qdt, dqdt, _ = sol.get_posvelacc(self.dt)
        qdt, dqdt = qdt[0], dqdt[0]
        self.prev_sol = sol
        self.prev_T = Tbest - self.dt
        
        # Log trajectory for animation
        t_full = np.linspace(0, Tbest, self.N_eval)
        q_traj, _, _ = sol.get_posvelacc(t_full)
        self.trajectory_log.append(q_traj)
        
        return qdt, dqdt


# =============================================================================
# Demo runner (keeps the look‑and‑feel of your SET script, but MPC inside)
# =============================================================================

def run_demo():
    q_min = np.zeros(2)
    q_max = 0.5*np.ones(2)
    env = CollisionEnvironment()

    q0 = np.array([0.15, 0.20])
    dq0 = np.zeros(2)
    q_goal = np.array([0.40, 0.30])

    ctrl = OnlineVPSTOMPC(
        q_goal=q_goal,
        q_min=q_min,
        q_max=q_max,
        env=env,
        ndof=2,
        N_eval=80,
        N_via=5,
        pop_size_warm=64,
        pop_size_explore=96,
        sigma_warm=0.08,
        sigma_explore=5.35,
        iters_per_tick_warm=6,
        iters_per_tick_explore=8,
        vel_lim=np.array([0.10, 0.10]),
        acc_lim=np.array([0.50, 0.50]),
        dt=0.05,
        hard_collision=1e6,
        hard_limits=1e6,
        soft_goal=1e3,
        smooth_w=1e-3,
    )

    # perfect tracking sim (like an impedance layer would do)
    Tsim = 12.0
    steps = int(Tsim/ctrl.dt)
    q_hist = [q0.copy()]
    dq_hist = [dq0.copy()]

    for _ in range(steps):
        q_ref, dq_ref = ctrl.step(q_hist[-1], dq_hist[-1])
        q_hist.append(q_ref.copy())
        dq_hist.append(dq_ref.copy())
        if np.linalg.norm(q_hist[-1] - q_goal) < 1e-2 and np.linalg.norm(dq_hist[-1]) < 1e-2:
            break

    q_hist = np.array(q_hist)
    dq_hist = np.array(dq_hist)
    t = np.arange(len(q_hist))*ctrl.dt

    # --- plots --------------------------------------------------------------
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2)

    # path
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlim(q_min[0], q_max[0])
    ax0.set_ylim(q_min[1], q_max[1])
    ax0.grid(True, alpha=0.3)
    env.draw(ax0)
    ax0.plot(q_hist[:, 0], q_hist[:, 1], 'b-', lw=2, label='MPC path')
    ax0.scatter(q_hist[0, 0], q_hist[0, 1], c='green', s=80, marker='o', label='Start')
    ax0.scatter(q_goal[0], q_goal[1], c='red', s=120, marker='*', label='Goal')
    ax0.legend(loc='lower right')
    ax0.set_title('VP‑STO MPC (full‑horizon optimize, short‑horizon track)')

    # velocity
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, dq_hist[:, 0], label='vx')
    ax1.plot(t, dq_hist[:, 1], label='vy')
    ax1.axhline(+ctrl.vel_lim[0], ls='--', c='r', lw=1)
    ax1.axhline(-ctrl.vel_lim[0], ls='--', c='r', lw=1)
    ax1.set_ylabel('velocity')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # distance to goal
    ax2 = fig.add_subplot(gs[1, 1])
    dist = np.linalg.norm(q_hist - q_goal[None, :], axis=1)
    ax2.plot(t, dist)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('‖q - q_goal‖')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Finished in {t[-1]:.2f}s, final error {dist[-1]:.4f}")

    # Create animation
    print("Creating animation...")
    create_animation(q_hist, dq_hist, ctrl, env, t, q0, q_goal, q_min, q_max)


def create_animation(q_hist, dq_hist, ctrl, env, time_vec, q0, q_goal, q_min, q_max):
    """Create and display animation of the MPC trajectory"""
    fps = 20
    robot_radius = 0.0001  # Visual radius for robot
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: trajectory animation
    ax1.set_facecolor('white')
    ax1.set_xlim(q_min[0] - 0.02, q_max[0] + 0.02)
    ax1.set_ylim(q_min[1] - 0.02, q_max[1] + 0.02)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('VP-STO MPC Real-time Trajectory')
    ax1.set_aspect('equal', adjustable='box')
    
    # Draw environment
    env.draw(ax1)
    
    # Create robot patch
    robot_patch = plt.Circle(q0, robot_radius, color='blue', alpha=0.5, zorder=10)
    ax1.add_patch(robot_patch)
    
    # Plot start and goal
    ax1.scatter(q0[0], q0[1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(q_goal[0], q_goal[1], c='red', s=120, marker='*', label='Goal', zorder=5)
    
    # Trajectory lines
    executed_line, = ax1.plot([], [], 'blue', linewidth=3, alpha=0.7, label='Executed path')
    
    # Planned trajectory line (current plan)
    planned_line, = ax1.plot([], [], 'magenta', linewidth=2, alpha=0.8, label='Current plan')
    
    # Candidate trajectory lines (sampled during optimization)
    max_candidates = max(ctrl.pop_warm, ctrl.pop_explore)  # Maximum number of candidates
    candidate_lines = []
    for i in range(min(max_candidates, 50)):  # Show up to 50 candidates for performance
        line, = ax1.plot([], [], 'orange', alpha=0.2, linewidth=0.8)
        candidate_lines.append(line)
    
    # Time and status text
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    status_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, fontsize=10, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add candidate trajectories to legend
    if candidate_lines:
        candidate_lines[0].set_label('Candidate trajectories')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: velocity profiles
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Velocity Profile')
    ax2.grid(True, alpha=0.3)
    
    # Velocity limit lines
    ax2.axhline(+ctrl.vel_lim[0], ls='--', c='r', lw=1, alpha=0.7, label='Velocity limits')
    ax2.axhline(-ctrl.vel_lim[0], ls='--', c='r', lw=1, alpha=0.7)
    
    # Velocity plots
    vx_line, = ax2.plot([], [], 'b-', label='vx', linewidth=2)
    vy_line, = ax2.plot([], [], 'r-', label='vy', linewidth=2)
    current_time_line = ax2.axvline(0, color='black', linestyle='-', alpha=0.7, label='Current time')
    
    ax2.legend()
    
    def init():
        executed_line.set_data([], [])
        planned_line.set_data([], [])
        for line in candidate_lines:
            line.set_data([], [])
        vx_line.set_data([], [])
        vy_line.set_data([], [])
        time_text.set_text('')
        status_text.set_text('')
        return [executed_line, planned_line] + candidate_lines + [vx_line, vy_line, robot_patch, time_text, status_text, current_time_line]
    
    def animate(frame):
        # Calculate current simulation step
        current_step = min(frame, len(q_hist) - 1)
        current_time = time_vec[current_step] if current_step < len(time_vec) else time_vec[-1]
        
        # Update robot position
        robot_patch.center = q_hist[current_step]
        
        # Update executed trajectory (path taken so far)
        if current_step > 0:
            executed_line.set_data(q_hist[:current_step+1, 0], q_hist[:current_step+1, 1])
        
        # Update planned trajectory (current MPC prediction)
        if current_step < len(ctrl.trajectory_log):
            planned_traj = ctrl.trajectory_log[current_step]
            planned_line.set_data(planned_traj[:, 0], planned_traj[:, 1])
        else:
            planned_line.set_data([], [])
        
        # Update candidate trajectories (from sampling)
        for line in candidate_lines:
            line.set_data([], [])  # Clear all lines first
            
        if current_step < len(ctrl.candidate_log) and ctrl.candidate_log[current_step] is not None:
            candidates = ctrl.candidate_log[current_step]
            candidate_positions = candidates['pos']  # Shape: (pop_size, N_eval, ndof)
            
            # Show a subset of candidates for better visualization
            n_show = min(len(candidate_lines), candidate_positions.shape[0], 80)  # Show up to 30 candidates
            
            for i in range(n_show):
                if i < len(candidate_lines):
                    # Use different colors/alpha based on candidate quality (optional enhancement)
                    candidate_lines[i].set_data(candidate_positions[i, :, 0], candidate_positions[i, :, 1])
                    candidate_lines[i].set_alpha(0.15 + 0.1 * (i == 0))  # Make first candidate slightly more visible
        
        # Update velocity plots
        if current_step > 0:
            vx_line.set_data(time_vec[:current_step+1], dq_hist[:current_step+1, 0])
            vy_line.set_data(time_vec[:current_step+1], dq_hist[:current_step+1, 1])
            
            # Update velocity plot limits
            ax2.set_xlim(0, max(time_vec[current_step] + 1, 2))
            max_vel = max(np.abs(dq_hist[:current_step+1]).max(), ctrl.vel_lim[0]) * 1.1
            ax2.set_ylim(-max_vel, max_vel)
            
        # Update current time line
        current_time_line.set_xdata([current_time, current_time])
        
        # Update text information
        time_text.set_text(f'Time: {current_time:.2f}s\nStep: {current_step}/{len(q_hist)-1}')
        
        # Status information
        speed = np.linalg.norm(dq_hist[current_step]) if current_step < len(dq_hist) else 0
        distance_to_goal = np.linalg.norm(q_hist[current_step] - q_goal)
        
        status_info = f'Speed: {speed:.3f} m/s\n'
        status_info += f'Dist to goal: {distance_to_goal:.4f} m'
        
        status_text.set_text(status_info)
        
        return [executed_line, planned_line] + candidate_lines + [vx_line, vy_line, robot_patch, time_text, status_text, current_time_line]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(q_hist), interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


if __name__ == "__main__":
    run_demo()
