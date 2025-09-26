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
start_pos = np.array([1, 1.5])   # 起始位置
goal_pos = np.array([1, -1])   # 目标位置
pos_traj = np.linspace(start_pos, goal_pos, 5)  # 生成直线路径点
# 计算对应的关节角度（选择肘部向下解）
joint_angles = np.array([robot.ik(pos)[0] for pos in pos_traj])
for q  in joint_angles:
    plotRobot(plt.gca(), robot, q, color='b')  # 绘制机械臂位置

plt.title('2D Manipulator Start and Goal')
plt.gca().set_aspect('equal')  # 设置坐标轴比例相等 
plt.legend()
plt.show()  # 显示绘图