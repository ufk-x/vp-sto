import numpy as np
from vpsto.vpsto import VPSTO,VPSTOOptions
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib import animation
import time
import datetime
class Manipulator():
    def __init__(self):
        self.l = np.array([1,1])
        self.q_min = np.array([-np.pi,-np.pi])
        self.q_max = np.array([np.pi,np.pi])
        self.r = 0.05
        self.q1_list = np.linspace(-np.pi,np.pi,200)

    def fk(self,q):
        x0 = np.zeros(2)
        x1 = x0 + self.l[0]*np.array([-np.sin(q[0]),np.cos(q[0])])
        x2 = x1 + self.l[1]*np.array([-np.sin(q[0]+q[1]), np.cos(q[0]+q[1])])
        return np.vstack((x0, x1, x2))
    
    
    def fk_(self,x0, q,l):
        x1 = x0 + l*np.array([-np.sin(q),np.cos(q)])
        return x1
    def collision_spheres(self, q):
        #返回所有用来碰撞检测的球体
        #每个连杆10个球
        x = self.fk(q)
        spheres = []
        for j in range(10):
            # s = self.fk_(x[0], q[0],(j*2+1)*self.r)
            # spheres.append(s)            
            s = self.fk_(x[1], q[0]+q[1],(j*2+1)*self.r)
            spheres.append(s)
        return np.array(spheres)

    def ik(self, x, y,l0,l1):
        r2 = x**2 + y**2
        cos_q1 = (r2 - l0**2 - l1**2) / (2*l0*l1)
        if abs(cos_q1) > 1.0:
            return None  # 无解
        q1_elbow_down = -np.arccos(cos_q1)
        q1_elbow_up = np.arccos(cos_q1)

        k1 = l0 + l1*np.cos(q1_elbow_down)
        k2 = l1*np.sin(q1_elbow_down)
        q0_elbow_down = np.arctan2(-x, y) - np.arctan2(k2, k1)

        k1 = l0 + l1*np.cos(q1_elbow_up)
        k2 = l1*np.sin(q1_elbow_up)
        q0_elbow_up = np.arctan2(-x, y) - np.arctan2(k2, k1)

        return (q0_elbow_down, q1_elbow_down), (q0_elbow_up, q1_elbow_up)

    def ik_(self, x, y,l0):
        if np.abs(x**2 + y**2 - l0**2)>0.1:
            return None
        sol = []
        q0_elbow_up = np.arctan2(x, y)
        for q in range(self.q1_list):
            q1_elbow_down = q
            sol.append(q0_elbow_up, q1_elbow_down)
        sol =np.array(sol)
        return sol

class CollisionEnvironment():
    def __init__(self):
        self.x0 = np.array([0.8,1.5])
        self.x = self.x0.copy()
        self.r = 0.1
        self.r_sq = self.r**2

        self.x_min = np.array([-1.2,-1])
        self.x_max = np.array([2.5,3.])
    def update_position(self, t):
        # 让障碍物沿X方向往返移动，比如正弦运动
        self.x[0] = self.x0[0] + 0.5 * np.sin(0.1*t)
        self.x[1] = self.x0[1] - 0.4 * np.cos(0.1*t)

    def isCollision(self, s, r):
        s = np.array(s)
        e1x = self.x - s#圆心到碰撞圆心点的向量
        d_sq = np.sum((e1x)**2,axis=1)#圆心到线段最短的距离平方# 按行求和：

        return d_sq<(self.r + r)**2  # 是否碰撞（距离平方 < 半径平方）axis=1

def plotEnvironment(ax, env):
    ax.set_xlim(env.x_min[0],env.x_max[0])
    ax.set_ylim(env.x_min[1],env.x_max[1])
    ax.set_aspect('equal')# 保证X和Y轴比例相同（避免图形拉伸）
    # 添加圆形障碍物（红色半透明圆）
    # ax.add_patch(patches.Circle(env.x, env.r, facecolor='r', edgecolor='None', alpha=0.5))

def plotRobot(ax, robot, q, color='k'):#'k'：黑色实线
    # 计算机械臂的正向运动学（返回关节点坐标）
    X = robot.fk(q)
    # 绘制连杆（黑色实线）
    ax.plot(X[:,0], X[:,1],'k')
    # 绘制关节（大圆点，颜色由参数决定）
    ax.plot(X[1:,0], X[1:,1], color+'o', markersize=6)  # 第2个点以后，用指定颜色画圆点

    # 绘制所有关节点（小黑点）
    # ax.plot(X[:,0], X[:,1], 'ko', markersize=4)  # 所有点用黑色('k')小圆点标记


robot = Manipulator()
env = CollisionEnvironment()
env.update_position(0)
opt = VPSTOOptions(ndof=2)  # 3自由度机械臂
opt.N_via = 3              # 路径点数量
opt.N_eval = 300             # 轨迹评估点数
opt.pop_size = 50          # 优化种群大小
opt.log = True              # 启用日志
opt.acc_lim = 10
opt.vel_lim =1
vpsto = VPSTO(opt)          # 初始化优化器
vpsto.opt.sigma_init = 8      # 初始化CMA-ES的步长
vpsto.opt.max_iter = 50    # 最大迭代次数
q0 = np.array([0.,0.])
dq0=np.array([0.,0.])
dqT = np.array([0.,0.])
xT = np.array([1.5,0.8])
t_local = 0.
sol_log = []
sim_duration = 5
dt_control = 0.1
sol1 = None
sol_his = []

print("x_max",env.x_max[0])
print("x_min",env.x_min[0])
def loss(candidates):
    global t_local
    costs = np.zeros(len(candidates['T']))#每个样本的代价
    for i in range(len(costs)):#每个样本 = 每条轨迹
        q_traj = candidates['pos'][i]
        q_lim_cost = (
            np.sum(np.maximum(q_traj[:,0] - robot.q_max[0], robot.q_min[0] - q_traj[:,0]) > 0) + 
            np.sum(np.maximum(q_traj[:,1] - robot.q_max[1], robot.q_min[1] - q_traj[:,1]) > 0)
            ) / (2*vpsto.opt.N_eval)
        # print("q_lim_cost",q_lim_cost)
         # 计算机械臂关节点坐标
        X = np.empty((len(q_traj),3,2))
        q_col_cost = 0.0
        # acc = 0
        for j in range(len(q_traj)):
            # print("q_traj",q_traj[j])
            X[j] = robot.fk(q_traj[j])
            env.update_position(t_local)
            s = robot.collision_spheres(q_traj[j])
            # print("X",X[j])
            q_col_cost += np.sum(env.isCollision(s,robot.r)) / (2 * vpsto.opt.N_eval)*10
            # acc += np.sum(candidates['acc'][i][j]**2)/(2 * vpsto.opt.N_eval)*10
        # print("q_col_cost",q_col_cost)        
        # 计算环境边界代价（超出x_min/x_max的惩罚）
        X_lim_cost = (
            np.sum(np.maximum(X[:,:,0] - env.x_max[0], env.x_min[0] - X[:,:,0]) > 0) +
            np.sum(np.maximum(X[:,:,1] - env.x_max[1], env.x_min[1] - X[:,:,1]) > 0)
        ) / (2*vpsto.opt.N_eval)
        
        np.set_printoptions(threshold=np.inf)
        end = X[-1,-1,:]
        de = np.sum((xT - end)**2)

        T = candidates['T'][i]

        costs[i] = T + 1e3 * (q_lim_cost + X_lim_cost +q_col_cost +de)

    return costs


def mpc(q,dq,sol = None):
    global t_local
    sol_ = sol
    if sol == None or sol.c_best>sol.T_best+10:
        sigma = 3
        p_init =None
    else:
        sigma = 0.1*sol.c_best/sol.T_best
        print("sol.c_best",sol.c_best)
        p_init = sol.p_best
    time1 = time.time()
    sol = vpsto.cma_trajectory(loss, q0=q, dq0=dq, dqT=dqT,sigma_init=sigma,p_init=p_init)
    print("t_local",t_local)
    print("ex",env.x)

    time2 = time.time()

    print("dt",time2 - time1)
    # 运行优化
    if sol_ is not None:
        if sol_.c_best < sol.c_best:
            sol = sol_
            print("111")
    p_best = sol.p_best
    T_best = sol.T_best
    print("T_best",T_best)  
    if T_best < dt_control:
        sol_log.append(np.vstack((xT,xT)))

        return p_best[-vpsto.opt.ndof:],np.zeros_like(dq)
    Tnext = T_best - dt_control
    t_next = np.linspace(0, Tnext, vpsto.opt.N_eval+1) + dt_control
    q_next, dq_next, _ = vpsto.vptraj.get_trajectory_at_time(t_next, p_best, q, dq0=dq, 
                                                            dqT=np.zeros_like(dq), T=T_best)
    X = np.empty((len(q_next), 2))
    print()
    for i in range(len(q_next)):
        y=robot.fk(q_next[i])
        # print("y",y)
        X[i] = y[-1,:]
        # print("y[-1,:]",y[-1,:])
        # print("X",X[i])
    sol_log.append(X)
    sol_his.append(sol)
    t_local+=dt_control
    print("q_next[1]",q_next[1])
    print("q_next[0]",q_next[0])
    return q_next[0], dq_next[0] ,sol



q_sim = [q0]
dq_sim = [dq0]

I = int(sim_duration / dt_control)
for i in range(I):
    result = mpc(q_sim[-1], dq_sim[-1], sol1)
    print("MPC returned:", len(result))  # 查看实际返回值的数量
    if len(result)<3:
        break
    q, dq, sol1 = result  # 确认无误后再解包
    q_sim.append(q.copy()) 
    dq_sim.append(dq.copy())
    
q_sim = np.array(q_sim)[1:]
dq_sim = np.array(dq_sim)[1:]
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

def video1():
    fps = 30

    # 创建图和轴
    fig = plt.figure(dpi=100)
    ax = plt.gca()
    ax.set_xlim(env.x_min[0]-0.5, env.x_max[0]+0.5)
    ax.set_ylim(env.x_min[1]-0.5, env.x_max[1]+0.5)
    ax.set_aspect('equal')

    # 绘制环境目标
    plotEnvironment(ax, env)
    plt.scatter(xT[0], xT[1], s=200, marker='*', color='gold',
                edgecolor='black', linewidth=0.5, zorder=10, label='Target') 

    # 最优轨迹线（品红色）
    pred_line, = ax.plot([], [], 'm', lw=2.5, label='Simulated trajectory', zorder=10)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='w', fontsize=14)

    # 连杆和关节
    link_line, = ax.plot([], [], 'k', animated=True)
    joint_dots, = ax.plot([], [], 'mo', markersize=6, animated=True)

    # 机械臂碰撞球
    sphere_r = 0.05
    num_spheres = 20  # 你collision_spheres生成球数量
    sphere_circles = [patches.Circle((0,0), radius=sphere_r, facecolor='g', alpha=0.5) for _ in range(num_spheres)]
    for c in sphere_circles:
        ax.add_patch(c)

    # 动态障碍物
    # collision_r = 0.3
    # print("env1111111.x",env.x)
    collision_circle = patches.Circle(env.x, radius=env.r, facecolor='r', alpha=0.5)
    ax.add_patch(collision_circle)

    # 初始化函数
    def init():
        pred_line.set_data([], [])
        time_text.set_text('')
        link_line.set_data([], [])
        joint_dots.set_data([], [])
        for c in sphere_circles:
            c.center = (0,0)
        collision_circle.center = env.x
        return [pred_line, time_text, link_line, joint_dots, collision_circle] + sphere_circles

    # 动画更新函数
    def animate(i_):
        i = np.min([len(q_sim)-1, int(i_ / (dt_control * fps))])
        t_sim = i * dt_control

        # 更新动态障碍物位置
        # print("t_sim",t_sim)
        env.update_position(t_sim)
        collision_circle.center = env.x
        # print("env.x",env.x)
        # print("env.x0",env.x0)

        # 更新最优轨迹
        pred_line.set_data(sol_log[i][:,0], sol_log[i][:,1])

        # 更新机械臂连杆与关节
        X = robot.fk(q_sim[i])
        x = robot.collision_spheres(q_sim[i])
        link_line.set_data(X[:,0], X[:,1])
        joint_dots.set_data(X[1:,0], X[1:,1])

        # 更新机械臂碰撞球
        for circle, pos in zip(sphere_circles, x):
            circle.center = pos

        # 更新时间显示
        time_text.set_text(f'time = {t_sim:.2f}s')

        return [pred_line, time_text, link_line, joint_dots, collision_circle] + sphere_circles

    # 创建动画
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=int(len(q_sim) * dt_control * fps),
                                   interval=1000.0/fps, blit=True)

    # 保存视频
    anim.save('arm_mpc_dynamic.mp4', fps=fps, codec='libx264')
video1()
# =======================
# 构建 C-space 网格
# =======================
n1_samples = 200
n2_samples = 200
q1 = np.linspace(-np.pi/2, np.pi/2, n1_samples)
q2 = np.linspace(-np.pi/2, np.pi/2, n2_samples)
c_space = np.zeros((n1_samples, n2_samples))
c_space_d = []
for i in range(n1_samples):
    for j in range(n2_samples):
        q = np.array([q1[i], q2[j]])
        X = robot.fk(q)
        # s = robot.collision_spheres(q)
        if np.any(q <= robot.q_min) or np.any(q >= robot.q_max):
            c_space[i, j] = 0
        elif np.any(X[:, 0] < env.x_min[0]) or np.any(X[:, 0] > env.x_max[0]) or np.any(X[:, 1] < env.x_min[1]) or np.any(X[:, 1] > env.x_max[1]):
            c_space[i, j] = 0
        else:
            c_space[i, j] = 1

# =======================
# 绘制函数（C-space + 速度/加速度 + 动画）
# =======================
def plot2DCSpaceBackGround():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(robot.q_min[0] - 0.5, robot.q_max[0] + 0.5)
    ax.set_ylim(robot.q_min[1] - 0.5, robot.q_max[1] + 0.5)
    ax.set_xlabel('Joint 1 (q1)')
    ax.set_ylabel('Joint 2 (q2)')
    ax.set_title('2D C-Space Visualization')
    X_, Y_ = np.where(c_space < 0.5)
    ax.scatter(q1[X_], q2[Y_], c='g', alpha=0.1, s=5, label='Obstacles')
    ax.scatter(q0[0], q0[1], c='r', s=50,marker='^',  label='Start')
    ax.scatter(q_sim[-1,0], q_sim[-1,1], c='b', s=50, marker='*', label='Via')
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
    # ax.scatter(sol.p_best[0], sol.p_best[1], c='b', s=50, label='via')
    ax.scatter(q_sim[-1, 0], q_sim[-1, 1], c='g', s=50, label='Goal')
    ax.plot(q_sim[:, 0], q_sim[:, 1], 'b-', linewidth=2, alpha=0.5, label='Trajectory')
    ax.legend(loc='upper right', 
              bbox_to_anchor=(1.0, 1.0),  # 精确定位
              framealpha=0.8,  # 背景透明度
              edgecolor='black',  # 边框颜色
              fontsize=10)  # 字体大小
    plt.tight_layout()

# 速度、加速度绘图

def plotvel_separate():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100, sharex=True)
    ax1.plot(q_sim[:, 0], 'b-', linewidth=1.5, label='Joint 1 Velocity')
    ax1.plot(q_sim[:, 1], 'g-', linewidth=1.5, label='Joint 2 Velocity')
    ax1.legend(loc='upper right')
    ax2.plot(dq_sim[:, 0], 'r-', linewidth=1.5, label='Joint 1 Acceleration')
    ax2.plot(dq_sim[:, 1], 'm-', linewidth=1.5, label='Joint 2 Acceleration')
    ax2.legend(loc='upper right')
    plt.show()
    return fig

# =======================
# 动画生成
# =======================
def update_cspace_obstacle(t_sim):
    # global c_space_d
    env.update_position(t_sim)
    # 假设 env.x 是动态障碍物位置
    obs_points = []
    # 球体上采样多个点
    for theta in np.linspace(-np.pi, np.pi, 400):
        px = env.x[0] + env.r*np.cos(theta)
        py = env.x[1] + env.r*np.sin(theta)
        
        ik_list = [
            robot.ik(px, py, robot.l[0], robot.l[1]),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.1),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.2),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.3),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.4),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.5),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.6),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.7),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.8),
            robot.ik(px, py, robot.l[0], robot.l[1]-0.9),
            robot.ik_(px, py, robot.l[0]),
            robot.ik_(px, py, robot.l[0]-0.1),
            robot.ik_(px, py, robot.l[0]-0.2),
            robot.ik_(px, py, robot.l[0]-0.3),
            robot.ik_(px, py, robot.l[0]-0.4),
            robot.ik_(px, py, robot.l[0]-0.5),
            robot.ik_(px, py, robot.l[0]-0.6),
            robot.ik_(px, py, robot.l[0]-0.7),
            robot.ik_(px, py, robot.l[0]-0.8),
            robot.ik_(px, py, robot.l[0]-0.9)
        ]

        for ik_sol in ik_list:
            if ik_sol is not None:
                obs_points.extend(ik_sol)
    obs_points = np.array(obs_points)
    # print("c_space_d",obs_points)
    # if len(obs_points) > 0:
    #     c_space_d=obs_points.copy()
    return obs_points

def video(his_sol):  
    fig = plot2DCSpaceBackGround()
    fps = 30
    ax = plt.gca()
    sample_lines = []
    pred_line, = ax.plot([], [], 'b', lw=2, label='Best Trajectory',zorder=10)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='k', fontsize=14)
    for i in range(100):
        line, = ax.plot([], [], 'orange', alpha=.25, label='Sample' if i == 0 else "")
        sample_lines.append(line)

    # 初始化 C-space 动态投影 scatter
    cspace_proj_scatter = ax.scatter([], [], c='k', s=5, alpha=0.2, label='Obstacle Projection')
    # pos = ax.scatter([], [], c='r', s=5, alpha=0.2, label='Obstacle Projection')
    pos = patches.Circle(env.x, radius=0.04, facecolor='r', alpha=0.5)
    ax.add_patch(pos)

    def init():
        pred_line.set_data([], [])
        time_text.set_text('')
        cspace_proj_scatter.set_offsets(np.empty((0,2)))
        pos.center=q0
        return time_text, pred_line, cspace_proj_scatter,pos

    def animate(i_):
        i = np.min([len(his_sol) - 1, int(i_ / (dt_control * fps))])
        for j in range(len(his_sol[i].history_pos_best)):
            sample_lines[j].set_data(his_sol[i].history_pos_best[j][:, 0], his_sol[i].history_pos_best[j][:, 1])
        time_text.set_text('time = %.1f' % (i * dt_control * 0.05))
        t_sim = i * dt_control
        # 更新动态障碍物投影到 C-space
        c_space_d = update_cspace_obstacle(t_sim) 
        # print("t_sim", t_sim)# 更新 c_space_d
        # print("c_space_d", c_space_d)# 更新 
        if len(c_space_d) > 0:
            all_proj = np.vstack(c_space_d)  # 合并所有帧投影点
            cspace_proj_scatter.set_offsets(all_proj)
        else:
            cspace_proj_scatter.set_offsets([])
        if i>0:
            pos.center=q_sim[i-1]
            traj,_,_=vpsto.vptraj.get_trajectory(his_sol[i].p_best,q_sim[i-1],dq0 = dq0,dqT=dqT,T=his_sol[i].T_best)

        else:
            pos.center=q0
            traj,_,_=vpsto.vptraj.get_trajectory(his_sol[i].p_best,q0,dq0 = dq0,dqT=dqT,T=his_sol[i].T_best)
        # print("traj",traj)
        pred_line.set_data(traj[0][:, 0], traj[0][:, 1])


        # 判断是否为最后一帧
        # is_last_frame = (i == len(his_sol) - 1)
        # 动态创建或更新点（只在最后一刻显示）
        if not hasattr(animate, 'final_point'):
            # 首次调用时创建点对象（初始状态隐藏）
            animate.final_point, = ax.plot([], [], 'rx', markersize=10, label='Via Point')
        # if is_last_frame:
        points = np.array(his_sol[-1].p_best).reshape(-1, 2) 
        # 最后一帧：显示点（位置取 his_best 的最后一个点）
        x, y = points[:, 0], points[:, 1]
        animate.final_point.set_data([x], [y])
        animate.final_point.set_visible(True)
        # else:
        #     # 非最后一帧：隐藏点
        #     animate.final_point.set_visible(False)
        ax.legend(loc='upper right')
        # 返回所有需要更新的对象（包括点）
        return (*sample_lines, pred_line, time_text, animate.final_point, pos)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=int(len(his_sol) * dt_control * fps),
                                   interval=1000.0 / fps, blit=True)
    anim.save(f'point_traj_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.mp4', fps=fps, codec='libx264')

# 执行绘图和视频生成
plot2DCSpace()
video(sol_his)
plotvel_separate()
