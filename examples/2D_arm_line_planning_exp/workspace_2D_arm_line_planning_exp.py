'''
Author: Fang Kai[thissfk@qq.com]
Date: 2025-09
LastEditors: Fang Kai[thissfk@qq.com]
LastEditTime: 2025-09
FilePath: workspace_2D_arm_line_planning_exp.py
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
# start_pos = np.array([0, 1])   # 起始位置；
goal_pos = np.array([1, -1])   # 目标位置
vel_lim = np.array([1, 1]) # 定义关节速度限制
acc_lim = np.array([0.5, 0.5]) # 定义关节加速度限制

start_q = robot.ik(start_pos)[0]  # 计算起始位置的关节角度（肘部向下解）
goal_q = robot.ik(goal_pos)[0]    # 计算目标位置的关节角度（肘部向下解）
# 初始化轨迹存储变量
q_profile = [start_q]  # 初始化关节角度轨迹
dq_profile = [np.zeros_like(start_q)]  # 初始化关节速度轨迹
ddq_profile = [np.zeros_like(start_q)]  # 初始化关节加速度轨迹

def dX2dq(dX, q):
    """
    计算末端速度与关节速度的雅可比矩阵
    参数：
    dX: 2x1数组，末端执行器速度
    q: 2x1数组，关节角度
    返回：2x2数组，雅可比矩阵
    """
    J = np.array([
        [-robot.l[0]*np.sin(q[0]) - robot.l[1]*np.sin(q[0]+q[1]), -robot.l[1]*np.sin(q[0]+q[1])],
        [ robot.l[0]*np.cos(q[0]) + robot.l[1]*np.cos(q[0]+q[1]),  robot.l[1]*np.cos(q[0]+q[1])]
    ])
    # 使用伪逆计算关节速度
    dq = np.linalg.pinv(J).dot(dX)
    return dq

h = 0.001 # 定义时间步长
q = start_q.copy()  # 初始化当前关节角度
dq_rec = 0
pos_tolerance = 1e-2  # 定义关节角度误差容忍度
iteration = 0  # 初始化迭代计数器
# 迭代更新关节角度，直到达到目标位置
while np.linalg.norm(goal_pos - robot.fk(q)[-1]) > pos_tolerance:
    print(f"迭代次数: {iteration}, 当前位置: {robot.fk(q)[-1]}, 目标位置: {goal_pos}, 位置误差: {np.linalg.norm(goal_pos - robot.fk(q)[-1])}")
    # 计算关节速度
    dX = (goal_pos - robot.fk(q)[-1]) / np.linalg.norm(goal_pos - robot.fk(q)[-1]) * 0.5  # 定义末端速度方向和大小
    dq = dX2dq(dX, q)
    ddq = (dq - dq_rec) / h  # 计算关节加速度
    ddq = np.clip(ddq, -acc_lim, acc_lim)  # 限制加速度
    dq = dq_rec + ddq * h  # 更新关节速度
    dq = np.clip(dq, -vel_lim, vel_lim)  # 限制速度
    dq_rec = dq.copy()  # 记录当前关节速度
    # 更新关节角度
    q += dq * h
    # 记录关节角度和速度
    q_profile.append(q.copy())
    dq_profile.append(dq.copy())
    iteration += 1  # 更新迭代计数器

# 打印数据
print(f"迭代次数: {iteration}, 轨迹点数: {len(q_profile)}")
q_profile = np.array(q_profile)      # 转换为numpy数组
dq_profile = np.array(dq_profile)    # 转换为numpy数组
ddq_profile = np.diff(dq_profile, axis=0) / h  # 计算加速度

# 绘制速度曲线
time_array = np.arange(len(dq_profile)) * h  # 计算时间数组
fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=100)  # 创建2行1列的子图
ax[0].plot(time_array, dq_profile[:,0], 'b-', label='DOF 1 velocity')  # 绘制第一个关节角度
ax[0].plot(time_array, dq_profile[:,1], 'r-', label='DOF 2 velocity')  # 绘制第二个关节角度
ax[0].axhline(-vel_lim[0],color = 'r',linestyle='-.',label='DOF 1 vel limit')
ax[0].axhline(vel_lim[0],color = 'r',linestyle='-.')
ax[0].axhline(vel_lim[1],color = 'r',linestyle='-.',label='DOF 2 vel limit')
ax[0].axhline(-vel_lim[1],color = 'r',linestyle='-.')
ax[0].set_title('Joint Velocities')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Velocity [rad/s]')
ax[0].legend()
ax[0].grid()
# 绘制加速度曲线
time_array_ddq = np.arange(len(ddq_profile)) * h  # 计算加速度时间数组
ax[1].plot(time_array_ddq, ddq_profile[:,0], 'b-', label='DOF 1 acceleration')  # 绘制第一个关节加速度
ax[1].plot(time_array_ddq, ddq_profile[:,1], 'r-', label='DOF 2 acceleration')  # 绘制第二个关节加速度
ax[1].axhline(acc_lim[0],color='r',linestyle='-.',label='DOF 1 acc limit')
ax[1].axhline(-acc_lim[0],color='r',linestyle='-.')
ax[1].axhline(acc_lim[1],color='r',linestyle='-.',label='DOF 2 acc limit')
ax[1].axhline(-acc_lim[1],color='r',linestyle='-.')
ax[1].set_title('Joint Accelerations')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Acceleration [rad/s²]')
ax[1].legend()
ax[1].grid()
plt.tight_layout()  # 自动调整子图间距
plt.savefig('./examples/2D_arm_line_planning_exp/media/workspace_line_planning_vel_acc.png')
print(f"saved as 'workspace_line_planning_vel_acc.png'")
plt.close()  # 关闭当前图形


# 描绘配置空间轨迹
fig, ax = plt.subplots()
ax.set_aspect('equal')
# 绘制关节角度轨迹
ax.plot(q_profile[:,0], q_profile[:,1], 'b-', label='q path')  # 绘制关节角度轨迹
ax.plot(start_q[0], start_q[1], 'go', markersize=8, label='start q')  # 绘制起始关节角度
ax.plot(goal_q[0], goal_q[  1], 'r*', markersize=8, label='goal q')  # 绘制目标关节角度
ax.set_title('Configuration Space Trajectory')
ax.set_xlabel('q1 [rad]')
ax.set_ylabel('q2 [rad]')
ax.grid()
ax.axis('equal')
ax.legend()
plt.savefig('./examples/2D_arm_line_planning_exp/media/workspace_line_planning_c_space_traj.png')
print(f"saved as 'workspace_line_planning_c_space_traj.png'")
plt.close()  # 关闭当前图形


# 绘制机械臂运动动画
def create_animation():
  import matplotlib.animation as animation
  fig, ax = plt.subplots(dpi=100)
  q_log = q_profile  # 使用之前计算的轨迹数据
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim(-2, 2)
  ax.set_ylim(-1.5, 1.5)

  # 绘制起始和目标配置
  # 末端执行器位置
  ax.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='start pos')  # 起始位置
  end_pos = goal_pos
  ax.plot(end_pos[0], end_pos[1], 'r*', markersize=12, label='goal pos')  # 目标位置
  ax.plot(start_pos,goal_pos, 'g--', alpha=0.3,label='line path')  # 连接起始和目标位置的线
  
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
  actual_duration = iteration * h  # 实际轨迹执行时间（秒）
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
    anim.save('./examples/2D_arm_line_planning_exp/media/workspace_line_planning_trajectory.mp4', 
              writer='ffmpeg', fps=video_fps, bitrate=1800)
    print(f"Animation saved as 'workspace_line_planning_trajectory.mp4' (fps: {video_fps:.1f})")
  except Exception as e:
    print(f"Could not save mp4 file: {e}")
    try:
        # 如果mp4失败，尝试gif
        gif_fps = min(10, total_frames / actual_duration)  # gif帧率限制
        anim.save('./examples/2D_arm_line_planning_exp/media/workspace_line_planning_trajectory.gif', 
                  writer='pillow', fps=gif_fps)
        print(f"Animation saved as 'workspace_line_planning_trajectory.gif' (fps: {gif_fps:.1f})")
    except Exception as e:
        print(f"Could not save gif file: {e}")
        print("Animation created but not saved. You can view it in the notebook.")
  
  plt.tight_layout()
  
  # 保存完成后立即关闭当前图形，释放资源
  plt.close(fig)
  
  return anim

# 调用函数创建动画
animation = create_animation()  # 创建并保存动画

# 清理资源，避免终端卡住
plt.close('all')  # 关闭所有matplotlib图形窗口