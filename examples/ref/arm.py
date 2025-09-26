import numpy as np
from vpsto.vpsto import VPSTO, VPSTOOptions
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import datetime
import os

# =======================
# 机械臂建模
# =======================
class Manipulator():
    def __init__(self):
        self.l = np.array([1, 1])  # 连杆长度
        self.q_min = np.array([-np.pi, -np.pi])  # 关节下限
        self.q_max = np.array([np.pi, np.pi])    # 关节上限

    def fk(self, q):
        """正向运动学，返回各个关节的坐标"""
        if q is None or len(q) < 2:
            return np.zeros((3, 2))
        x0 = np.zeros(2)
        x1 = x0 + self.l[0] * np.array([-np.sin(q[0]), np.cos(q[0])])
        x2 = x1 + self.l[1] * np.array([-np.sin(q[0] + q[1]), np.cos(q[0] + q[1])])
        return np.vstack((x0, x1, x2))

# =======================
# 环境建模（障碍物 + 边界）
# =======================
class CollisionEnvironment():
    def __init__(self):
        self.x = np.array([0.7, 1.3])  # 圆形障碍物中心
        self.r = 0.1                   # 半径
        self.r_sq = self.r**2

        self.x_min = np.array([-1.0, -0.5])
        self.x_max = np.array([2., 2.])

    def isTrajectoryCollision(self, pts):
        pts_ = np.empty((pts.shape[0] * (pts.shape[1] - 1), 4))
        pts_[:, :2] = pts[:, :-1].reshape(-1, 2)
        pts_[:, 2:] = pts[:, 1:].reshape(-1, 2)
        collisions_over_time = np.any(self.isCollision(pts_).reshape(pts.shape[0], pts.shape[1] - 1), axis=1)
        return collisions_over_time

    def isRobotCollision(self, pts):
        pts_ = np.empty((pts.shape[0] - 1, 4))
        pts_[:, :2] = pts[:-1]
        pts_[:, 2:] = pts[1:]
        return np.any(self.isCollision(pts_))

    def isCollision(self, pts):
        e12 = pts[:, 2:] - pts[:, :2]
        e1x = self.x - pts[:, :2]
        lam = np.clip(np.sum(e12 * e1x, axis=1) / np.sum(e12**2, axis=1), 0, 1)
        d_sq = np.sum((e1x - (lam * e12.T).T)**2, axis=1)
        return d_sq < self.r_sq

# =======================
# 绘制函数
# =======================
def plotEnvironment(ax, env):
    ax.set_xlim(env.x_min[0], env.x_max[0])
    ax.set_ylim(env.x_min[1], env.x_max[1])
    ax.set_aspect('equal')
    ax.add_patch(patches.Circle(env.x, env.r, facecolor='r', edgecolor='None', alpha=0.5, label="Obstacle"))

def plotRobot(ax, robot, q, color='k'):
    X = robot.fk(q)
    ax.plot(X[:, 0], X[:, 1], 'k')
    ax.plot(X[1:, 0], X[1:, 1], color + 'o', markersize=6)

# =======================
# 优化设置
# =======================
robot = Manipulator()
env = CollisionEnvironment()

opt = VPSTOOptions(ndof=2)
opt.N_via = 2
opt.N_eval = 100
opt.pop_size = 100
opt.log = True
opt.sigma_init = 8
opt.max_iter = 100
vpsto = VPSTO(opt)

q0 = np.array([0., 0.])
dq0 = np.array([0., 0.])
dqT = np.array([0., 0.])
xT = np.array([0.9, 0.5])
# qT = np.array([-0.03155091, -2.06186755])

# =======================
# 成本函数
# =======================
def loss(candidates):
    costs = np.zeros(len(candidates['T']))
    for i in range(len(costs)):
        q_traj = candidates['pos'][i]
        q_lim_cost = (
            np.sum(np.maximum(q_traj[:, 0] - robot.q_max[0], robot.q_min[0] - q_traj[:, 0]) > 0) +
            np.sum(np.maximum(q_traj[:, 1] - robot.q_max[1], robot.q_min[1] - q_traj[:, 1]) > 0)
        ) / (2 * vpsto.opt.N_eval)

        X = np.empty((len(q_traj), 3, 2))
        acc = 0
        for j in range(len(q_traj)):
            X[j] = robot.fk(q_traj[j])
            acc += np.sum(candidates['acc'][i][j]**2)

        X_lim_cost = (
            np.sum(np.maximum(X[:, :, 0] - env.x_max[0], env.x_min[0] - X[:, :, 0]) > 0) +
            np.sum(np.maximum(X[:, :, 1] - env.x_max[1], env.x_min[1] - X[:, :, 1]) > 0)
        ) / (2 * vpsto.opt.N_eval)

        q_col_cost = np.sum(env.isTrajectoryCollision(X)) / (2 * vpsto.opt.N_eval)

        end = X[-1, -1, :]
        de = np.sum((xT - end)**2)
        # de = 0  # 关闭终点误差约束

        T = candidates['T'][i]
        costs[i] = T + 1000.0 * (q_lim_cost + X_lim_cost + q_col_cost + de) + acc

    return costs

# =======================
# 运行优化
# =======================
sol = vpsto.cma_trajectory(loss, q0=q0, dq0=dq0,  dqT=dqT)
q_via_opt = sol.p_best

# =======================
# 绘制函数（Cartesian-space + 轨迹）
# =======================
def plotCartesianSpace(it):
    plt.figure(dpi=100)
    ax = plt.gca()    
    # ===== 1. 设置坐标轴范围和刻度 =====
    ax.set_xlim(-0.5,2.0)  # X轴显示范围
    ax.set_ylim(-0.5, 2.5)  # Y轴显示范围
    
    # 设置主刻度（显示数值标签）
    ax.set_xticks(np.linspace(-1.0,2.0, 7)) # X轴从-1.5到1.5分7个刻度
    ax.set_yticks(np.linspace(-0.5, 2.0, 6))  # Y轴同理
    
    # # 设置次刻度（不显示标签）
    # # 次刻度设置（需要先设置主刻度）
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))  # 次刻度间隔0.5
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    
    # 网格线设置（必须放在刻度设置之后）
    ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    
    # 坐标轴标签
    ax.set_xlabel("X Position (m)", fontsize=10)
    ax.set_ylabel("Y Position (m)", fontsize=10)

    # 绘制环境和机械臂
    plotEnvironment(ax, env)                # 绘制障碍物和边界
    plotRobot(ax, robot, q0, color='r')     # 初始位置（红色）

    # 选择当前最优路径点
    if it == 0:
        q_via_best = q0
        plotRobot(ax, robot, q_via_best, color='m')  # 路径点（品红色）

    else:
        print("pbest1",sol.p_best)
        print("pbest2",sol.p_best[-2:])
        q_via_best = sol.p_best  # 优化后的路径点
        for i in range(len(q_via_best)//2):
            print("q_via_best",q_via_best[i*2:(i+1)*2])
            plotRobot(ax, robot, q_via_best[i*2:(i+1)*2], color='m')  # 路径点（品红色）
            # plotRobot(ax, robot, qT, color='m')  # 路径点（品红色）
        
    plt.scatter(xT[0], xT[1], s=200, marker='*', color='gold', edgecolor='black', linewidth=0.5, zorder=10, label='Target') 
    # 生成并绘制轨迹
    q_traj, _, _ = vpsto.vptraj.get_trajectory(sol.p_best, q0,dq0=dq0,dqT= dqT)
    q_traj = q_traj.squeeze(axis=0)
    X = np.empty((len(q_traj), 3, 2))
    for i in range(len(q_traj)):
        X[i] = robot.fk(q_traj[i])  # 计算机械臂轨迹的关节点坐标
    plt.plot(X[:,-1,0], X[:,-1,1], c='m', alpha=0.8, label='arm trajectory')  # 绘制末端轨迹
    plt.legend(loc='upper right')
    
    # 确保输出目录存在
    output_dir = 'media/czq_ref'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/task_space_{it}.png', dpi=200)

# =======================
# 构建 C-space 网格
# =======================
n1_samples = 200
n2_samples = 200
q1 = np.linspace(robot.q_min[0], robot.q_max[0], n1_samples)
q2 = np.linspace(robot.q_min[1], robot.q_max[1], n2_samples)
c_space = np.zeros((n1_samples, n2_samples))
for i in range(n1_samples):
    for j in range(n2_samples):
        q = np.array([q1[i], q2[j]])
        X = robot.fk(q)
        if np.any(q <= robot.q_min) or np.any(q >= robot.q_max):
            c_space[i, j] = 0
        elif np.any(X[:, 0] < env.x_min[0]) or np.any(X[:, 0] > env.x_max[0]) or np.any(X[:, 1] < env.x_min[1]) or np.any(X[:, 1] > env.x_max[1]):
            c_space[i, j] = 0
        elif env.isRobotCollision(robot.fk(q)):
            c_space[i, j] = 0
        else:
            c_space[i, j] = 1

# =======================
# 绘制函数（C-space + 速度/加速度 + 动画）
# =======================
def plot2DCSpaceBackGround():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(robot.q_min[0], robot.q_max[0])
    ax.set_ylim(robot.q_min[1], robot.q_max[1])
    ax.set_xlabel('Joint 1 (q1)')
    ax.set_ylabel('Joint 2 (q2)')
    ax.set_title('2D C-Space Visualization')
    X_, Y_ = np.where(c_space < 0.5)
    ax.scatter(q1[X_], q2[Y_], c='g', alpha=0.1, s=5, label='Obstacles')
    ax.scatter(q0[0], q0[1], c='r', s=50, label='Start')
    ax.scatter(sol.p_best[-2], sol.p_best[-1], c='b', s=50, label='Via')
    # ax.scatter(sol.p_best[0], sol.p_best[1], c='b', s=50, label='Via')
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot2DCSpace():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)  # 2D 坐标轴
    
    # 设置坐标范围
    ax.set_xlim(robot.q_min[0], robot.q_max[0])
    ax.set_ylim(robot.q_min[1], robot.q_max[1])
    ax.set_xlabel('Joint 1 (q1)')
    ax.set_ylabel('Joint 2 (q2)')
    ax.set_title('2D C-Space Visualization')
    ax.xaxis.set_major_locator(plt.AutoLocator())  # 自动选择刻度间隔
    ax.yaxis.set_major_locator(plt.AutoLocator())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # 次刻度间隔0.5
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(which='major', linestyle='-', linewidth=0.5, color='k', alpha=0.5)  # 主网格线
    ax.grid(which='minor', linestyle='--', linewidth=0.1, color='k', alpha=0.8)  # 次网格线（虚线）
    # 绘制障碍物（假设 c_space 是 2D 数组）
    X, Y = np.where(c_space < 0.5)  # 障碍物阈值
    ax.scatter(q1[X], q2[Y], c='k', alpha=0.1, s=5, label='Obstacles')

    # 绘制最优轨迹
    ax.scatter(q0[0], q0[1], c='r', s=50, label='Start')
    ax.scatter(sol.p_best[0], sol.p_best[1], c='b', s=50, label='via')
    ax.scatter(q_traj[-1, -1, 0], q_traj[-1, -1, 1], c='g', s=50, label='Goal')
    ax.plot(q_traj[0,:, 0], q_traj[0,:, 1], 'b-', linewidth=2, alpha=0.5, label='Trajectory')
    ax.legend(loc='upper right', 
              bbox_to_anchor=(1.0, 1.0),  # 精确定位
              framealpha=0.8,  # 背景透明度
              edgecolor='black',  # 边框颜色
              fontsize=10)  # 字体大小
    plt.tight_layout()
    
    # 保存图片到指定目录
    output_dir = 'media/czq_ref'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/c_space.png', dpi=200)
    plt.show()

# 速度、加速度绘图
q_traj, dq_traj, ddq_traj = vpsto.vptraj.get_trajectory(sol.p_best, q0, dq0=dq0,dqT=dqT, T=sol.T_best)
dq_traj = dq_traj[0]
ddq_traj = ddq_traj[0]

def plotvel_separate():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100, sharex=True)
    ax1.plot(dq_traj[:, 0], 'b-', linewidth=1.5, label='Joint 1 Velocity')
    ax1.plot(dq_traj[:, 1], 'g-', linewidth=1.5, label='Joint 2 Velocity')
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend(loc='upper right')
    ax2.plot(ddq_traj[:, 0], 'r-', linewidth=1.5, label='Joint 1 Acceleration')
    ax2.plot(ddq_traj[:, 1], 'm-', linewidth=1.5, label='Joint 2 Acceleration')
    ax2.set_ylabel('Acceleration (rad/s²)')
    ax2.set_xlabel('Time Steps')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    
    # 保存图片到指定目录
    output_dir = 'media/czq_ref'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/velocity_acceleration.png', dpi=200)
    plt.show()
    return fig

# =======================
# 动画生成
# =======================
history_array = np.array(sol.history)
his_best = np.array(sol.history_pos_best)

def video():
    import os
    
    # 确保输出目录存在
    output_dir = 'media/czq_ref'
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查历史数据
    if len(history_array) == 0:
        print("Warning: No history data available for animation")
        return
    
    fig = plot2DCSpaceBackGround()
    fps = 30
    dt_control = 0.1
    ax = plt.gca()
    sample_lines = []
    pred_line, = ax.plot([], [], 'b', lw=1.5, label='Best Trajectory')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='k', fontsize=14)
    
    for i in range(len(history_array[0])):
        line, = ax.plot([], [], 'orange', alpha=.25, label='Sample' if i == 0 else "")
        sample_lines.append(line)

    def init():
        pred_line.set_data([], [])
        time_text.set_text('')
        for line in sample_lines:
            line.set_data([], [])
        return (*sample_lines, pred_line, time_text)

    def animate(i_):
        i = np.min([len(history_array) - 1, int(i_ / (dt_control * fps))])
        
        # 更新采样轨迹
        for j in range(min(len(history_array[i]), len(sample_lines))):
            sample_lines[j].set_data(history_array[i][j][:, 0], history_array[i][j][:, 1])
        
        # 更新最优轨迹
        pred_line.set_data(his_best[i][:, 0], his_best[i][:, 1])
        time_text.set_text('time = %.1f' % (i * dt_control * 0.05))
        
        # 判断是否为最后一帧
        is_last_frame = (i == len(history_array) - 1)
        
        # 动态创建或更新点（只在最后一刻显示）
        if not hasattr(animate, 'final_point'):
            animate.final_point, = ax.plot([], [], 'rx', markersize=10, label='Via Point')
        
        if is_last_frame:
            points = np.array(sol.p_best).reshape(-1, 2) 
            x, y = points[:, 0], points[:, 1]
            animate.final_point.set_data([x], [y])
            animate.final_point.set_visible(True)
        else:
            animate.final_point.set_visible(False)
        
        ax.legend(loc='upper right')
        return (*sample_lines, pred_line, time_text, animate.final_point)

    # 创建动画
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=int(len(history_array) * dt_control * fps),
                                   interval=1000.0 / fps, blit=True)
    
    # 生成文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    mp4_filename = f'{output_dir}/point_traj_{timestamp}.mp4'
    gif_filename = f'{output_dir}/point_traj_{timestamp}.gif'
    
    # 优先尝试保存为 MP4
    try:
        # 使用 ffmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='VP-STO'), bitrate=1800, 
                       extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        anim.save(mp4_filename, writer=writer)
        print(f"MP4 视频已保存到: {mp4_filename}")
    except Exception as e:
        print(f"MP4 保存失败: {e}")
        try:
            # 备选方案：保存为 GIF
            anim.save(gif_filename, writer='pillow', fps=15)
            print(f"GIF 动画已保存到: {gif_filename}")
        except Exception as e2:
            print(f"GIF 保存也失败: {e2}")
            print("将直接显示动画...")
            plt.show()
    
    plt.close(fig)

# 执行绘图和视频生成
print("开始生成 Cartesian space 图片...")
plotCartesianSpace(100)
print("Cartesian space 图片生成完成")

print("开始生成 C-space 图片...")
plot2DCSpace()
print("C-space 图片生成完成")

print("开始生成速度/加速度图...")
plotvel_separate()
print("速度/加速度图生成完成")

print(f"历史数据长度: {len(history_array)}")
print(f"最优历史数据长度: {len(his_best)}")

print("开始生成视频...")
video()
print("所有任务完成")

# 备用视频生成函数（如果上面的失败）
def video_alternative():
    """备用视频生成方法 - 使用不同的编码器设置"""
    import os
    
    output_dir = 'media/czq_ref'
    os.makedirs(output_dir, exist_ok=True)
    
    if len(history_array) == 0:
        print("Warning: No history data available for animation")
        return
    
    fig = plot2DCSpaceBackGround()
    fps = 20  # 降低帧率
    ax = plt.gca()
    sample_lines = []
    pred_line, = ax.plot([], [], 'b', lw=1.5, label='Best Trajectory')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='k', fontsize=14)
    
    for i in range(len(history_array[0])):
        line, = ax.plot([], [], 'orange', alpha=.25, label='Sample' if i == 0 else "")
        sample_lines.append(line)

    def animate(frame):
        i = min(len(history_array) - 1, frame)
        for j in range(min(len(history_array[i]), len(sample_lines))):
            sample_lines[j].set_data(history_array[i][j][:, 0], history_array[i][j][:, 1])
        pred_line.set_data(his_best[i][:, 0], his_best[i][:, 1])
        time_text.set_text(f'Iteration: {i}')
        return (*sample_lines, pred_line, time_text)

    anim = animation.FuncAnimation(fig, animate, frames=len(history_array), 
                                   interval=1000//fps, blit=False, repeat=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 尝试不同的保存方法
    success = False
    
    # 方法1: 简化的 MP4 (优先选择)
    if not success:
        try:
            filename = f'{output_dir}/point_traj_{timestamp}.mp4'
            anim.save(filename, writer='ffmpeg', fps=fps)
            print(f"MP4已保存到: {filename}")
            success = True
        except Exception as e:
            print(f"MP4保存失败: {e}")
    
    # 方法2: GIF
    if not success:
        try:
            filename = f'{output_dir}/point_traj_{timestamp}.gif'
            anim.save(filename, writer='pillow', fps=fps)
            print(f"GIF已保存到: {filename}")
            success = True
        except Exception as e:
            print(f"GIF保存失败: {e}")
    
    # 方法3: HTML
    if not success:
        try:
            filename = f'{output_dir}/point_traj_{timestamp}.html'
            anim.save(filename, writer='html', fps=fps)
            print(f"HTML动画已保存到: {filename}")
            success = True
        except Exception as e:
            print(f"HTML保存失败: {e}")
    
    if not success:
        print("所有保存方法都失败，显示动画...")
        plt.show()
    
    plt.close(fig)
