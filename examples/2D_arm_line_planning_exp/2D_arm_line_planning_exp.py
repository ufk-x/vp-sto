'''
Author: Fang Kai[thissfk@qq.com]
Date: 2025-09
LastEditors: Fang Kai[thissfk@qq.com]
LastEditTime: 2025-09
FilePath: 2D_arm_line_planning_exp.py
Description: 
           If you need more information,
please contact Fang Kai[thissfk@qq.com] to get an access.   
Copyright (c) 2025 by Fang Kai, All Rights Reserved. 
'''
# 导入必要的库
import numpy as np                          # 数值计算库
from vpsto.vpsto import VPSTO, VPSTOOptions # VPSTO轨迹优化库
from vpsto.obf import OBF                   # 椭球基函数库
import matplotlib.pyplot as plt             # 绘图库
import matplotlib.patches as patches        # matplotlib几何图形补丁
import matplotlib.animation as animation

PI = np.pi  # 定义π常量

# 定义2D机械臂类，包含正向运动学计算
class Manipulator():
    def __init__(self):
        # 机械臂链接长度：两段链接长度均为1
        self.l = np.array([1, 1]) # link lengths
        # 关节角度限制：第一关节[0, π]，第二关节[-π, 0]
        self.q_min = np.array([0., -np.pi])    # 关节角度下限
        self.q_max = np.array([np.pi, 0.])     # 关节角度上限
        self.danger_dist = 5 * 1e-2  # 定义安全距离

    # 定义机械臂的正向运动学
    def fk(self, q):
        """
        正向运动学函数：根据关节角度计算机械臂各关节的笛卡尔坐标位置
        参数：
        q: 2x1数组，包含两个关节的角度
        返回：3x2数组，包含基座、第一关节和末端执行器的二维位置
        """
        x0 = np.zeros(2)  # 基座位置（原点）
        # 第一关节位置：基座 + 第一段链接的笛卡尔位置
        x1 = x0 + self.l[0] * np.array([np.cos(q[0]), np.sin(q[0])])
        # 末端执行器位置：第一关节 + 第二段链接的笛卡尔位置
        x2 = x1 + self.l[1] * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
        return np.vstack((x0, x1, x2))  # 返回所有关节的位置
    
    def ik(self, pos):
        """
        逆向运动学函数：根据末端执行器位置计算关节角度
        参数：
        pos: 2x1数组，包含末端执行器的二维位置
        返回：2x1数组，包含两个关节的角度
        """
        x, y = pos  # 提取末端执行器的x和y坐标
        l1, l2 = self.l  # 提取链接长度
        # 计算第二关节角度（肘部角度）
        cos_q2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        sin_q2_elbow_down = -np.sqrt(1 - cos_q2**2)  # 选择肘部向下的解
        sin_q2_elbow_up = np.sqrt(1 - cos_q2**2)  # 选择肘部向上的解
        q2_elbow_up = np.arctan2(sin_q2_elbow_up, cos_q2)
        q2_elbow_down = np.arctan2(sin_q2_elbow_down, cos_q2)
        # 计算第一关节角度（肩部角度）
        k1 = l1 + l2 * cos_q2
        k2_elbow_up = l2 * sin_q2_elbow_up
        k2_elbow_down = l2 * sin_q2_elbow_down
        q1_elbow_up = np.arctan2(y, x) - np.arctan2(k2_elbow_up, k1)
        q1_elbow_down = np.arctan2(y, x) - np.arctan2(k2_elbow_down, k1)

        return np.array([q1_elbow_down, q2_elbow_down]),np.array([q1_elbow_up, q2_elbow_up])  # 返回关节角度
    
def plotRobot(ax, robot, q, color='m'):
    """
    在给定坐标轴上绘制机械臂
    参数：
    ax: matplotlib坐标轴对象
    robot: Manipulator实例
    q: 关节角度数组
    color: 绘制颜色
    """
    # 通过正向运动学计算机械臂各关节位置
    positions = robot.fk(q)
    # 绘制机械臂链接线（黑色线条）
    ax.plot(positions[:,0], positions[:,1], 'k-')
    # 绘制关节点（彩色圆圈，除基座外）
    ax.plot(positions[1:-1,0], positions[1:-1,1], color+'o', markersize=6,alpha=0.3)
    ax.plot(positions[-1,0], positions[-1,1], color+'o', markersize=6)  # 末端执行器
    # 绘制机械臂周围危险区域
    for i in range(positions.shape[0]-1):
        p1 = positions[i]
        p2 = positions[i+1]
        # 为每个关节添加危险区域圆圈
        joint1_circle = patches.Circle(p1, robot.danger_dist, color='red', alpha=0.3)
        joint2_circle = patches.Circle(p2, robot.danger_dist, color='red', alpha=0.3)
        ax.add_patch(joint1_circle)
        ax.add_patch(joint2_circle)
        # 计算线段方向向量
        e12 = p2 - p1
        # 计算法向量（垂直于线段）
        n = np.array([-e12[1], e12[0]]) # 和e12点乘为0
        n = n / np.linalg.norm(n) * robot.danger_dist  # 归一化并缩放到安全距离大小
        # 计算线段两侧的平行线端点
        p1_left = p1 + n
        p1_right = p1 - n
        p2_left = p2 + n
        p2_right = p2 - n
        # 画出危险区域（用多边形填充）
        # label = 'arm zone' if i==0 else None
        label =  None
        polygon = patches.Polygon([p1_left, p2_left, p2_right, p1_right], color='red', alpha=0.3,label=label)
        ax.add_patch(polygon)

"""创建机械臂实例并测试逆向运动学"""
# mp = Manipulator()  # 创建机械臂实例
# positions = np.array([1,1])  # 定义末端执行器位置
# q_down, q_up = mp.ik(positions)  # 计算逆向运动学，获取关节角度
# print(f"肘部向下关节角度： {q_down}\n")  # 打印关节角度
# print(f"肘部向上关节角度： {q_up}\n")  # 打印关节角度
# fig, ax = plt.subplots()  # 创建绘图窗口和坐标轴
# ax.set_aspect('equal')  # 设置坐标轴比例相等
# ax.set_xlim(-0.1, 2.1)  # 设置x轴范围
# ax.set_ylim(-1.1, 1.1)  # 设置y轴范围
# # plotRobot(ax, mp, q_down, color='b')  # 绘制肘部向下的机械臂
# plotRobot(ax, mp, q_up, color='g')    # 绘制肘部向上的机械臂
# ax.plot(positions[0], positions[1], 'ro', markersize=8, label='target pos')  # 绘制目标位置
# ax.set_title('2D Manipulator IK')
# ax.legend() 
# plt.show()  # 显示绘图

# 创建机械臂实例
robot = Manipulator()
# 定义起始和目标末端位置
start_pos = np.array([1, 1])   # 起始位置
goal_pos = np.array([1, -1])   # 目标位置
start_q = robot.ik(start_pos)[0]  # 计算起始位置的关节角度（肘部向下解）
goal_q = robot.ik(goal_pos)[0]    # 计算目标位置的关节角度（肘部向下解）

# 定义使得末端移动最短的损失函数
def loss(candidates):
    costs = []
    q_logs = candidates['pos']
    # 计算每个候选解末端行走的路程
    for i in range(q_logs.shape[0]): 
        dis = 0
        q_log = q_logs[i]
        end_log = [robot.fk(q)[2] for q in q_log]  # 计算每个时间步的末端位置
        for j in range(len(end_log)-1):
            dis += np.linalg.norm(end_log[j+1]-end_log[j])  # 累加末端位置间的距离
        costs.append(dis)  # 将总距离作为代价
    return np.array(costs)  # 返回所有候选解的代价数组

"""测试损失函数"""
# pos_log = np.linspace(start_pos, goal_pos, 5)  # 生成从起始到目标的线性插值轨迹
# print(pos_log)
# q_log = [robot.ik(pos)[0] for pos in pos_log]  # 计算每个位置对应的关节角度（肘部向下解）
# fig, ax = plt.subplots()  # 创建绘图窗口和坐标轴
# for q in q_log:
#     plotRobot(ax, robot, q, color='b')  # 绘制轨迹上的机械臂位置
# ax.set_aspect('equal')  # 设置坐标轴比例相等
# plt.show()  # 显示绘图
# candidates = {'pos': np.array([q_log])}  # 创建候选解字典
# cost = loss(candidates)  # 计算损失
# print(f"Cost of straight line in joint space: {cost[0]}")  # 打印损失值

# 设置VPSTO优化选项
opt = VPSTOOptions(ndof=2)
opt.vel_lim = np.array([1, 1])
opt.acc_lim = np.array([0.5, 0.5])
# opt.acc_lim = np.array([2.5, 2.5])
opt.N_via = 4
opt.N_eval = 100
opt.pop_size = 25
opt.max_iter = 200
opt.sigma_init = 1.5
traj_opt = VPSTO(opt)
# 进行轨迹优化
sol = traj_opt.minimize(loss, start_q, qT=goal_q, dqT=np.zeros_like(goal_q))
# 提取优化结果并绘图
t_traj = np.linspace(0, sol.T_best, 1000)
q_log, dq_log, ddq_log = sol.get_posvelacc(t_traj)

# 绘制速度曲线
plt.subplot(2,1,1)
plt.plot(t_traj, dq_log[:, 0], 'b-', label='DOF 1 velocity')
plt.plot(t_traj, dq_log[:, 1], 'r-', label='DOF 2 velocity')
plt.axhline(-opt.vel_lim[0], color='r', linestyle='-.', label='DOF 1 vel limit') # 负方向限制线
plt.axhline(opt.vel_lim[1], color='r', linestyle='-.', label='DOF 2 vel limit')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [rad/s]')
plt.title('Joint Velocities')
plt.legend()
plt.grid()
# 绘制加速度曲线
plt.subplot(2,1,2)
plt.plot(t_traj, ddq_log[:, 0], 'b-', label='DOF 1 acceleration')
plt.plot(t_traj, ddq_log[:, 1], 'r-', label='DOF 2 acceleration')
plt.axhline(-opt.acc_lim[0], color='r', linestyle='-.', label='DOF 1 acc limit') # 负方向限制线
plt.axhline(opt.acc_lim[1], color='r', linestyle='-.', label='DOF 2 acc limit')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [rad/s²]')
plt.title('Joint Accelerations')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./examples/2D_arm_line_planning_exp/media/line_planning_vel_acc.png', dpi=300)
print("Velocity and acceleration profiles saved as 'line_planning_vel_acc.png'")
# 绘制配置空间轨迹
fig, ax = plt.subplots()
for q in q_log:
    plt.plot(q[0], q[1], 'bo', markersize=2)  # 绘制轨迹点
plt.plot(start_q[0], start_q[1], 'go', markersize=10, label='Start')  # 起始点
plt.plot(goal_q[0], goal_q[1], 'r*', markersize=10, label='Goal')    # 目标点
plt.title('Configuration Space Trajectory')
plt.xlabel('Joint 1 Angle [rad]')
plt.ylabel('Joint 2 Angle [rad]')
plt.legend()
plt.grid()
plt.axis('equal')
plt.savefig('./examples/2D_arm_line_planning_exp/media/line_planning_config_space.png', dpi=300)
print("Configuration space trajectory saved as 'line_planning_config_space.png'")
# 绘制机械臂运动动画
def create_animation():
  import matplotlib.animation as animation
  fig, ax = plt.subplots(dpi=100)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim(-0.1, 2.1)
  ax.set_ylim(-1.1, 1.1)

  # 绘制起始和目标配置
  # 末端执行器位置
  ax.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='start pos')  # 起始位置
  end_pos = goal_pos
  ax.plot(end_pos[0], end_pos[1], 'r*', markersize=12, label='goal pos')  # 目标位置
  
  # 计算所有末端执行器位置用于轨迹显示
  X_all = np.empty((len(q_log), 3, 2))
  for k in range(len(q_log)):
    X_all[k] = robot.fk(q_log[k])
  
  # 初始化动画元素
  robot_line, = ax.plot([], [], 'k-', linewidth=2)
  robot_joints, = ax.plot([], [], 'mo', markersize=6)
  end_effector_trail, = ax.plot([], [], 'm-', alpha=0.7, linewidth=1, label='end-effector trail')
  
  ax.legend()
  ax.set_title('2D Manipulator Trajectory Animation')
  
  # 记录初始的patches数量（障碍物等固定元素）
  initial_patches_count = len(ax.patches)
  
  def animate(frame):
    # 当前机械臂配置
    positions = X_all[frame]
    
    # 更新机械臂链接线
    robot_line.set_data(positions[:,0], positions[:,1])
    
    # 更新关节点
    robot_joints.set_data(positions[1:,0], positions[1:,1])

    # 清除之前动态添加的危险区域patches（保留固定的环境障碍物）
    # 只保留初始的patches（障碍物等固定元素）
    while len(ax.patches) > initial_patches_count:
        ax.patches[-1].remove()
    
    # 绘制机械臂危险区域
    for i in range(positions.shape[0]-1):
        p1 = positions[i]
        p2 = positions[i+1]
        
        # 为每个关节添加危险区域圆圈
        joint1_circle = patches.Circle(p1, robot.danger_dist, color='red', alpha=0.3)
        joint2_circle = patches.Circle(p2, robot.danger_dist, color='red', alpha=0.3)
        ax.add_patch(joint1_circle)
        ax.add_patch(joint2_circle)
        
        # 计算线段方向向量
        e12 = p2 - p1
        # 计算法向量（垂直于线段），和e12点乘为0
        n = np.array([-e12[1], e12[0]])  
        if np.linalg.norm(n) > 0:  # 避免除零错误
            n = n / np.linalg.norm(n) * robot.danger_dist  # 归一化并缩放到安全距离大小
            # 计算线段两侧的平行线端点
            p1_left = p1 + n
            p1_right = p1 - n
            p2_left = p2 + n
            p2_right = p2 - n
            # 画出危险区域（用多边形填充）
            polygon = patches.Polygon([p1_left, p2_left, p2_right, p1_right], 
                                    color='red', alpha=0.3)
            ax.add_patch(polygon)
    
    # 更新末端执行器轨迹
    if frame > 0:
      trail_x = X_all[:frame+1, -1, 0]
      trail_y = X_all[:frame+1, -1, 1]
      end_effector_trail.set_data(trail_x, trail_y)
    
    return robot_line, robot_joints, end_effector_trail
  
  # 计算动画参数以匹配实际轨迹时间
  total_frames = len(q_log)
  actual_duration = sol.T_best  # 实际轨迹执行时间（秒）
  fps = 30  # 设置帧率为30fps
  
  # 计算每帧的时间间隔（毫秒）
  frame_interval = (actual_duration * 1000) / total_frames
  
  print(f"轨迹实际执行时间: {actual_duration:.2f} 秒")
  print(f"动画总帧数: {total_frames}")
  print(f"动画帧率: {fps} fps")
  print(f"动画总时长: {actual_duration:.2f} 秒")
  
  # 创建动画
  anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                  interval=frame_interval, blit=False, repeat=True)

  # 尝试保存为不同格式 - 优先保存mp4
  try:
    # 首先尝试保存为mp4格式，使用计算出的fps确保时间匹配
    video_fps = total_frames / actual_duration  # 根据实际时间计算fps
    anim.save('./examples/2D_arm_line_planning_exp/media/line_planning_trajectory.mp4', 
              writer='ffmpeg', fps=video_fps, bitrate=1800)
    print(f"Animation saved as 'line_planning_trajectory.mp4' (fps: {video_fps:.1f})")
  except Exception as e:
    print(f"Could not save mp4 file: {e}")
    try:
        # 如果mp4失败，尝试gif
        gif_fps = min(10, total_frames / actual_duration)  # gif帧率限制
        anim.save('./examples/2D_arm_line_planning_exp/media/line_planning_trajectory.gif', 
                  writer='pillow', fps=gif_fps)
        print(f"Animation saved as 'line_planning_trajectory.gif' (fps: {gif_fps:.1f})")
    except Exception as e:
        print(f"Could not save gif file: {e}")
        print("Animation created but not saved. You can view it in the notebook.")
  
  plt.tight_layout()
  return anim

# 调用函数创建动画
animation = create_animation()  # 创建并保存动画