import numpy as np
from vpsto.vpsto import VPSTO,VPSTOOptions
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib import animation
import time
class Manipulator():
    def __init__(self):
        self.l = np.array([1,1])
        self.q_min = np.array([-np.pi/2,-np.pi/2])
        self.q_max = np.array([np.pi/2,np.pi/2])
        self.r = 0.05

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
            s = self.fk_(x[0], q[0],(j*2+1)*self.r)
            spheres.append(s)            
            s = self.fk_(x[1], q[0]+q[1],(j*2+1)*self.r)
            spheres.append(s)
        return np.array(spheres)
class CollisionEnvironment():
    def __init__(self):
        self.x = np.array([1.,1.3])
        self.r = 0.1
        self.r_sq = self.r**2

        self.x_min = np.array([-1.2,-1])
        self.x_max = np.array([2.5,3.])

    def isCollision(self, s, r):
        s = np.array(s)
        e1x = self.x - s#圆心到碰撞圆心点的向量
        d_sq = np.sum((e1x)**2,axis=1)#圆心到线段最短的距离平方# 按行求和：
        # print("s",s)
        # print("x",self.x)
        # print("d_sq:",d_sq)
        # print("e1x:",e1x)
        # print("CCCC",d_sq<(self.r_sq + r**2))
        return d_sq<(self.r_sq + r**2)  # 是否碰撞（距离平方 < 半径平方）axis=1

def plotEnvironment(ax, env):
    ax.set_xlim(env.x_min[0],env.x_max[0])
    ax.set_ylim(env.x_min[1],env.x_max[1])
    ax.set_aspect('equal')# 保证X和Y轴比例相同（避免图形拉伸）
    # 添加圆形障碍物（红色半透明圆）
    ax.add_patch(patches.Circle(env.x, env.r, facecolor='r', edgecolor='None', alpha=0.5))

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


print("x_max",env.x_max[0])
print("x_min",env.x_min[0])
def loss(candidates):
    costs = candidates['T']
    '''
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
        acc = 0
        for j in range(len(q_traj)):
            # print("q_traj",q_traj[j])
            X[j] = robot.fk(q_traj[j])
            s = robot.collision_spheres(q_traj[j])
            # print("X",X[j])
            q_col_cost += np.sum(env.isCollision(s,robot.r)) / (2 * vpsto.opt.N_eval)*10
            acc += np.sum(candidates['acc'][i][j]**2)/(2 * vpsto.opt.N_eval)*10
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
        costs[i] = T + 1e3 * (q_lim_cost + X_lim_cost +q_col_cost +de+acc)  
        '''

    return costs

sol_log = []
sim_duration = 3
dt_control = 0.3
sol1 = None
def mpc(q,dq,sol = None):
    sol_ = sol
    if sol == None or sol.c_best>sol.T_best+10 :
        sigma = 8
        p_init =None
    else:
        sigma = 4*sol.c_best/sol.T_best
        print("sol.c_best",sol.c_best)
        p_init=sol.p_best
    time1 = time.time()
    sol= vpsto.cma_trajectory(loss, q0=q, dq0=dq, dqT=dqT,sigma_init=sigma,p_init=p_init)
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

        return p_best[-vpsto.opt.ndof:],np.zeros_like(dq),sol
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
    # print("q_next[1]",q_next[1])
    # print("q_next[0]",q_next[0])
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

# print("q_sim",q_sim)
fps=30
fig=plt.figure(dpi=100)
ax = plt.gca()

# 绘制环境和目标
plotEnvironment(ax, env)                # 绘制障碍物和边界
plt.scatter(xT[0], xT[1], s=200, marker='*', color='gold', edgecolor='black', linewidth=0.5, zorder=10, label='Target') 

# 选择当前最优路径点
    


# 最优轨迹线（品红色）
pred_line, = ax.plot([], [], 'm', lw=2.5, label='Simulated trajectory', zorder=10)

# 时间显示文本
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='w', fontsize=14)

# 候选轨迹线（绿色半透明）
sample_lines = []
link_line, = ax.plot([], [], 'k', animated=True)  # 连杆
joint_dots, = ax.plot([], [], 'mo', markersize=6, animated=True)  # 关节
sphere_dots, = ax.plot([], [], 'go', markersize=6, animated=True)  # 关节
def init():
    pred_line.set_data([], [])  # 清空最优轨迹
    time_text.set_text('')      # 清空时间文本
    link_line.set_data([], [])
    joint_dots.set_data([], [])
    sphere_dots.set_data([], [])
    return pred_line, time_text, link_line, joint_dots,sphere_dots

def animate(i_):
    # 将动画帧数转换为仿真步数
    i = np.min([len(q_sim)-1, int(i_ / (dt_control * fps))])
    
    # 更新最优轨迹显示
    pred_line.set_data(sol_log[i][:,0], sol_log[i][:,1])
    
    # 更新机械臂位置
    X = robot.fk(q_sim[i])
    x = robot.collision_spheres(q_sim[i])
    link_line.set_data(X[:,0], X[:,1])
    joint_dots.set_data(X[1:,0], X[1:,1])
    sphere_dots.set_data(x[1:,0],x[1:,1])
    # 更新时间显示
    time_text.set_text('time = %.1f' % (i*dt_control*0.05))
    # if i%5==0:
    #     print("X",X)
    #     print("x",x)

    return pred_line, time_text, link_line, joint_dots,sphere_dots

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(len(q_sim) * dt_control * fps), interval=1e3/fps, blit=True)

anim.save('arm_mpc.mp4', fps=fps, codec='libx264')

