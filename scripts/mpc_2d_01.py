import numpy as np

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import animation

from config import PLANNER_PATH
sys.path.append(PLANNER_PATH)
from planners.vptraj import VPTraj

# Parameters
q0 = np.array([-.5, -.5])  # Initial position
qg = np.array([.5, .5])    # Goal position
dq0 = np.array([0, 0])     # Initial velocity
dqg = np.array([0, 0])     # Goal velocity

bounds = 0.8*np.array([[-1, 1], [-1, 1]])  # Bounds on position

R = 1e1        # Penalty on control effort (acceleration)
Q_min = 1e0    # Minimum penalty on control error (position)
Q_max = 1e3    # Maximum penalty on control error (position)
factor_Q_min = 1e-1
factor_Q_max = 1e1

N_via = 4        # Number of via points
N_candidates = 100  # Number of candidates to sample
N_eval = 25      # Number of pos,vel,acc samples to evaluate along each candidate trajectory

vel_lim = 0.2    # Velocity limit (m/s in each dimension)
acc_lim = 1      # Acceleration limit (m/s^2 in each dimension)

num_obstacles = 10    # Number of obstacles
robot_radius = 0.1    # Radius of robot (m)
obstacle_radius = 0.1 # Radius of obstacle (m)

dt_control = 0.05     # Time step for control (s)
sim_duration = 30     # Duration of simulation (s)


class Obstacle:
    def __init__(self, pos, vel, radius, robot_radius):
        self.pos = pos  # initial position
        self.vel = vel  # initial velocity
        self.radius = radius  # radius of obstacle
        self.robot_radius = robot_radius  # radius of robot
        self.d_sq = (radius + robot_radius + 0.01)**2  # distance squared for faster collision checking

        self.history = [self.pos.copy()]  # history of obstacle positions

    def reset(self):
        self.history = [self.pos.copy()]

    def step(self, dt):
        # Check if obstacle would bounce off a wall and reverse velocity if so
        for i in range(2):
            if self.pos[i] + self.vel[i] * dt < bounds[i, 0] + self.radius:
                self.vel[i] = -self.vel[i]
            if self.pos[i] + self.vel[i] * dt > bounds[i, 1] - self.radius:
                self.vel[i] = -self.vel[i]
        self.pos += self.vel * dt
        self.history.append(self.pos.copy())

    def is_collision(self, robot_pos):
        # Check if the robot at pos is in collision with the obstacle
        # pos is a (N+1)D array of shape (M1, ..., MN, 2)
        # Returns a ND array of shape (M1, ..., MN) containing True if there is a collision
        return np.sum((robot_pos - self.pos)**2, axis=-1) < self.d_sq
    
    def predict_collision(self, robot_pos_traj, T):
        # Predict if there will be a collision in the next T seconds
        # pos_traj is a 2D array of shape (N, 2) containing the discretized trajectory of the robot
        # T is a scalar
        # Returns the number of collisions in the next T seconds
        N = robot_pos_traj.shape[0]
        t_traj = np.linspace(0, T, N)
        obs_traj = self.pos + self.vel * t_traj[:, None]
        return np.sum((robot_pos_traj - obs_traj)**2, axis=-1) < self.d_sq
    
    def predict_collision_batch(self, robot_pos_traj, T):
        # Predict if there will be a collision in the next T seconds
        # pos_traj is a 3D array of shape (N_batch, N, 2) containing the discretized trajectories of the robot
        # T is a vector of shape (N_batch,)
        # Returns the number of collisions in the next T seconds for each batch element
        N_batch, N, _ = robot_pos_traj.shape
        t_traj = np.linspace(0, T, N)
        obs_traj = self.pos + np.swapaxes(self.vel * t_traj[:, :, None], 0, 1)  # (N_batch, N, 2)
        return np.sum((robot_pos_traj - obs_traj)**2, axis=-1) < self.d_sq


class PredictiveSamplingController:
    def __init__(self, N_eval, N_via, vel_lim, acc_lim, bounds, qg, obstacles, dt_control, N_candidates, R):
        self.vptraj = VPTraj(ndof=2, N_eval=N_eval, N_via=N_via, vel_lim=vel_lim, acc_lim=acc_lim)
        self.vptraj_idle = VPTraj(ndof=2, N_eval=N_eval, N_via=1, vel_lim=vel_lim, acc_lim=acc_lim)

        self.bounds = bounds  # Bounds on position
        self.qg = qg  # Goal position
        self.obstacles = obstacles  # List of obstacles to avoid
        self.dt_control = dt_control  # Time step for control
        self.N_candidates = N_candidates  # Number of candidate trajectories to sample
        self.R = R  # Acceleration penalty at sampling stage (not considered in loss function)
        self.Q = Q_max  # Bias towards trajectories that go closer to the goal (updated at each iteration)
        self.acc_lim = acc_lim  # Acceleration limit (used for constructing idle trajectory)

        self.p_next = None  # Candidate trajectory parameter for next iteration
        self.T_next = None  # Candidate trajectory duration for next iteration
        self.samples_log = []  # Log of all candidate trajectories
        self.samples_loss_log = []  # Log of all candidate trajectories' losses
        self.sol_log = []  # Log of all solutions
        self.Q_log = []  # Log of all Q values

    def reset(self):
        self.p_next = None
        self.T_next = None
        self.samples_log = []
        self.samples_loss_log = []
        self.sol_log = []
        self.Q_log = []

    # Loss function for the candidate trajectories
    def loss_fn(self, q, dq, ddq, T):
        # Penalize trajectory duration
        duration_cost = T  # 时间代价
        # Penalize control error
        qT = q[:, -1]  # Final position of all candidates
        terminal_cost = 1e3 * np.sum((qT - self.qg)**2, axis=1)  # Squared error，终端误差
        # Penalize position limit violations (soft constraint)
        num_violations = np.sum((q < self.bounds[:, 0] + robot_radius) | (q > self.bounds[:, 1] - robot_radius), axis=(1, 2))  # 越界惩罚
        limit_violation_cost = 1e6 * num_violations
        # Penalize collisions with the obstacles (soft constraint)
        collision_cost = 0
        for obs in self.obstacles:
            collision_cost += 1e6 * np.sum(obs.predict_collision_batch(q, T), axis=1)  # 碰撞惩罚

        return terminal_cost + limit_violation_cost + duration_cost + collision_cost
    
    # Control function that samples candidate trajectories and chooses the best one
    def predictive_sampling(self, q, dq):
        # Compute how many constraint violations have occurred in the previous iteration
        if len(self.samples_loss_log) > 0:
            num_violations = np.sum(self.samples_loss_log[-1] > 1e6)
        else:
            num_violations = 0
        # Compute how strong the bias should be towards the goal:
        # - If there have been few constraint violations, increase the bias
        # - If there have been many constraint violations, decrease the bias
        self.Q *= np.clip(np.exp(-3 * (num_violations/self.N_candidates-0.5)), factor_Q_min, factor_Q_max)
        self.Q = np.clip(self.Q, Q_min, Q_max)
        self.Q_log.append(self.Q)

        # Sample candidate trajectories, compute their loss and return the best one
        pos, vel, acc, p, T = self.vptraj.sample_trajectories(self.N_candidates, q, dq0=dq, qT=self.qg, 
                                                              dqT=np.zeros_like(dq), Q=self.Q, R=self.R)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], T)
        self.samples_log.append(pos)
        self.samples_loss_log.append(loss)
        i_best = np.argmin(loss)
        return p[i_best], loss[i_best], T[i_best]
    
    # Control function that reuses the previous solution
    def previous_sol(self, q, dq):
        if self.p_next is None:
            return None, np.inf, 0
        # Compute trajectory with previous solution
        pos, vel, acc = self.vptraj.get_trajectory(self.p_next, q, dq0=dq, dqT=np.zeros_like(dq), T=self.T_next)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [self.T_next])[0]
        return self.p_next, loss, self.T_next
    
    # Control function that makes the robot come to stop as fast as possible
    def idle(self, q, dq):
        # Assuming constant acceleration
        T_idle = np.max(np.abs(dq) / self.acc_lim)
        # Compute the stopping position
        q_idle = q + 0.5 * dq * T_idle
        pos, vel, acc = self.vptraj_idle.get_trajectory(q_idle, q, dq0=dq, dqT=np.zeros_like(dq), T=T_idle)
        loss = self.loss_fn(pos[:, 1:], vel[:, 1:], acc[:, 1:], [T_idle])[0]
        # Check obstacle collision also for two seconds ahead
        q_list = np.vstack((q, q_idle))
        loss += 1e6 * np.sum([obs.predict_collision(q_list, T_idle + 2) for obs in self.obstacles])
        t_next = np.min([T_idle, self.dt_control])
        if t_next < dt_control:
            return q_idle, np.zeros_like(dq), loss, 0
        q_next, dq_next, _ = self.vptraj_idle.get_trajectory_at_time(t_next, q_idle, q, dq0=dq, dqT=np.zeros_like(dq), T=T_idle)
        return q_next.squeeze(), dq_next.squeeze(), loss, T_idle
    
    def control(self, q, dq):
        # Compute idle trajectory
        q_next_idle, dq_next_idle, loss_idle, T_idle = self.idle(q, dq)
        # Compute trajectory with previous solution
        p_prev, loss_prev, T_prev = self.previous_sol(q, dq)
        # Compute trajectory with predictive sampling
        p_samp, loss_samp, T_samp = self.predictive_sampling(q, dq)


        # Choose the best trajectory
        if loss_idle < loss_prev and loss_idle < loss_samp:  # 如果空闲轨迹损失最小，通常是因为机器人已经接近目标点
            print(f"Idle: {loss_idle:.2f}", end='\r')
            self.p_next = None
            self.T_next = None
            loss_best = loss_idle
            self.sol_log.append(np.vstack((q, q_next_idle)))
            return q_next_idle, dq_next_idle
        elif loss_prev < loss_samp:
            print(f"Previous: {loss_prev:.2f}", end='\r')
            p_best = p_prev
            T_best = T_prev
            loss_best = loss_prev
        else:
            print(f"Sampling: {loss_samp:.2f}", end='\r')
            p_best = p_samp
            T_best = T_samp
            loss_best = loss_samp

        if T_best < self.dt_control:
            self.p_next = None
            self.T_next = None
            self.sol_log.append(np.vstack((q, q)))
            return p_best[-self.vptraj.ndof:], np.zeros_like(dq)
        # Compute trajectory parameter and duration for next time step
        self.T_next = T_best - self.dt_control
        t_next = np.linspace(0, self.T_next, self.vptraj.N_via+1) + self.dt_control
        q_next, dq_next, _ = self.vptraj.get_trajectory_at_time(t_next, p_best, q, dq0=dq, 
                                                                dqT=np.zeros_like(dq), T=T_best)
        self.p_next = q_next[1:].flatten()
        self.sol_log.append(q_next)

        return q_next[0], dq_next[0]


def main():
    print("Initializing obstacles and controller...")
    
    # Initialize obstacles at random positions and velocities
    obstacles = []
    for i in range(num_obstacles):
        valid = False
        while not valid:
            pos = np.random.uniform(bounds[:, 0] + obstacle_radius, bounds[:, 1] - obstacle_radius)
            vel = np.random.uniform(-.1, .1, size=2)
            obs = Obstacle(pos, vel, obstacle_radius, robot_radius)
            valid = not obs.is_collision(q0)
        obstacles.append(obs)

    controller = PredictiveSamplingController(N_eval, N_via, vel_lim, acc_lim, bounds, qg, 
                                            obstacles, dt_control, N_candidates, R)

    print(f"Running simulation for {sim_duration} seconds...")
    
    # Simulate closed-loop system for sim_duration seconds
    q_sim = [q0]
    dq_sim = [dq0]

    I = int(sim_duration / dt_control)
    for i in range(I):
        if i % 50 == 0:  # Progress indicator
            print(f"Progress: {i/I*100:.1f}%")
        
        for obs in obstacles:
            obs.step(dt_control)  # Simulate obstacle motion
        
        q, dq = controller.control(q_sim[-1], dq_sim[-1])  # Compute next desired position and velocity
        # As we pretend we have a perfect low-level controller, we can directly apply the desired state
        q_sim.append(q.copy()) 
        dq_sim.append(dq.copy())

    q_sim = np.array(q_sim)[1:]
    dq_sim = np.array(dq_sim)[1:]

    print("\nSimulation complete. Generating plots...")

    # Visualize results
    time_vec = np.linspace(0, sim_duration, len(q_sim))
    
    # Plot position trajectories
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(time_vec, q_sim)
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Robot Position vs Time')
    plt.legend(['x', 'y'])
    plt.grid(True)

    # Plot velocity trajectories
    plt.subplot(2, 2, 2)
    plt.plot(time_vec, dq_sim)
    # plot the velocity limits
    plt.plot([0, sim_duration], [vel_lim, vel_lim], 'r--', label='Velocity limit')
    plt.plot([0, sim_duration], [-vel_lim, -vel_lim], 'r--')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.title('Robot Velocity vs Time')
    plt.legend(['vx', 'vy'])
    plt.grid(True)

    # Plot Q values
    plt.subplot(2, 2, 3)
    plt.plot(np.linspace(0, sim_duration, len(controller.Q_log)), np.log10(controller.Q_log))
    plt.xlabel('Time (s)')
    plt.ylabel('log10(Q)')
    plt.title('Q Parameter Evolution')
    plt.grid(True)

    # Plot 2D trajectory
    plt.subplot(2, 2, 4)
    plt.plot(q_sim[:, 0], q_sim[:, 1], 'b-', linewidth=2, label='Robot trajectory')
    plt.scatter(q0[0], q0[1], c='green', s=100, marker='o', label='Start', zorder=5)
    plt.scatter(qg[0], qg[1], c='red', s=100, marker='x', label='Goal', zorder=5)
    
    # Plot obstacles initial positions
    for i, obs in enumerate(obstacles):
        circle = plt.Circle(obs.history[0], obs.radius, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
        if i == 0:  # Only add label once
            circle.set_label('Obstacles')
    
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('2D Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSimulation Summary:")
    print(f"Final position: [{q_sim[-1, 0]:.3f}, {q_sim[-1, 1]:.3f}]")
    print(f"Distance to goal: {np.linalg.norm(q_sim[-1] - qg):.3f}")
    print(f"Average Q value: {np.mean(controller.Q_log):.2e}")
    print(f"Max velocity: {np.max(np.linalg.norm(dq_sim, axis=1)):.3f}")
    
    # Check for collisions
    total_collisions = 0
    for obs in obstacles:
        for q_t in q_sim:
            if obs.is_collision(q_t):
                total_collisions += 1
    print(f"Total collision instances: {total_collisions}")

    # Create animation
    print("Creating animation...")
    create_animation(q_sim, dq_sim, controller, obstacles, time_vec)


def create_animation(q_sim, dq_sim, controller, obstacles, time_vec):
    """Create and display animation of the robot trajectory"""
    fps = 20
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel('X Position', color='white')
    ax.set_ylabel('Y Position', color='white')
    ax.set_title('Robot Navigation with Predictive Sampling', color='white')
    ax.tick_params(colors='white')
    
    # Create obstacle patches
    obstacle_patches = []
    for obs in obstacles:
        patch = plt.Circle(obs.history[0], obs.radius, color='red', alpha=0.7)
        ax.add_patch(patch)
        obstacle_patches.append(patch)
    
    # Create robot patch
    robot_patch = plt.Circle(q0, robot_radius, color='cyan', alpha=0.8)
    ax.add_patch(robot_patch)
    
    # Plot start and goal
    ax.scatter(q0[0], q0[1], c='green', s=150, marker='o', label='Start', zorder=10)
    ax.scatter(qg[0], qg[1], c='yellow', s=150, marker='*', label='Goal', zorder=10)
    
    # Trajectory line
    trajectory_line, = ax.plot([], [], 'cyan', linewidth=2, alpha=0.7, label='Robot path')
    
    # Predicted trajectory lines (for current candidates)
    candidate_lines = []
    for i in range(min(50, N_candidates)):  # Show only first 50 for performance
        line, = ax.plot([], [], 'green', alpha=0.3, linewidth=1)
        candidate_lines.append(line)
    
    # Selected trajectory line
    selected_line, = ax.plot([], [], 'magenta', linewidth=3, alpha=0.8, label='Selected trajectory')
    
    # Time and status text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white', 
                       fontsize=12, verticalalignment='top')
    status_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, color='white', 
                         fontsize=10, verticalalignment='top')
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    def init():
        trajectory_line.set_data([], [])
        selected_line.set_data([], [])
        for line in candidate_lines:
            line.set_data([], [])
        time_text.set_text('')
        status_text.set_text('')
        return [trajectory_line, selected_line] + candidate_lines + [time_text, status_text]
    
    def animate(frame):
        # Calculate current simulation step
        current_step = min(frame, len(q_sim) - 1)
        current_time = time_vec[current_step]
        
        # Update robot position
        robot_patch.center = q_sim[current_step]
        
        # Update obstacle positions
        for i, obs in enumerate(obstacles):
            if current_step < len(obs.history):
                obstacle_patches[i].center = obs.history[current_step]
        
        # Update trajectory (path taken so far)
        if current_step > 0:
            trajectory_line.set_data(q_sim[:current_step+1, 0], q_sim[:current_step+1, 1])
        
        # Update candidate trajectories (if available)
        if current_step < len(controller.samples_log):
            samples = controller.samples_log[current_step]
            losses = controller.samples_loss_log[current_step]
            
            # Normalize losses for color coding
            if len(losses) > 0:
                min_loss = np.min(losses)
                max_loss = np.max(losses)
                if max_loss > min_loss:
                    normalized_losses = (losses - min_loss) / (max_loss - min_loss)
                else:
                    normalized_losses = np.zeros_like(losses)
                
                # Show subset of candidates
                n_show = min(len(candidate_lines), samples.shape[0])
                for i in range(n_show):
                    if i < samples.shape[0]:
                        # Color based on performance (green=good, red=bad)
                        color_val = 1 - normalized_losses[i]
                        color = (1-color_val, color_val, 0)  # Red to green
                        candidate_lines[i].set_data(samples[i, :, 0], samples[i, :, 1])
                        candidate_lines[i].set_color(color)
                        candidate_lines[i].set_alpha(0.4)
                    else:
                        candidate_lines[i].set_data([], [])
        
        # Update selected trajectory
        if current_step < len(controller.sol_log):
            sol = controller.sol_log[current_step]
            selected_line.set_data(sol[:, 0], sol[:, 1])
        
        # Update text information
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        # Status information
        speed = np.linalg.norm(dq_sim[current_step]) if current_step < len(dq_sim) else 0
        distance_to_goal = np.linalg.norm(q_sim[current_step] - qg)
        
        status_info = f'Speed: {speed:.3f} m/s\n'
        status_info += f'Distance to goal: {distance_to_goal:.3f} m\n'
        
        if current_step < len(controller.Q_log):
            status_info += f'Q parameter: {controller.Q_log[current_step]:.1e}'
        
        status_text.set_text(status_info)
        
        return [trajectory_line, selected_line] + candidate_lines + obstacle_patches + [robot_patch, time_text, status_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(q_sim), interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


if __name__ == "__main__":
    main()
