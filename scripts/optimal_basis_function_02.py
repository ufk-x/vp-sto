"""
这个脚本的目的是展示如何使用最优基函数 (Optimal Basis Function, OBF) 来生成轨迹。

基函数本身只取决于节点数和维度数，与具体的节点位置无关。
因此基函数可以预先计算并存储，以便在需要时快速调用。

在这个例子中，我们将展示如何生成二维轨迹，并可视化基函数、轨迹、速度和加速度。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from config import PROJECT_DIR
sys.path.append(PROJECT_DIR)

from planners.obf import OBF

def main(figsize=(18, 12)):
    N_via = 3
    N_eval = 100
    h = np.full(N_via, 1.0 / N_via)  # shape: [N_via] 每个节点之间的时间间隔
    obf = OBF(ndof=2)
    obf.setup_task(h)
    s = np.linspace(0, np.sum(h), N_eval)  # 考虑相位
    phi = obf.get_Phi(s)        # shape: [len(s), N_via+3] 获取基函数
    dphi = obf.get_dPhi(s)      # shape: [len(s), N_via+3] 获取一阶导数基函数
    ddphi = obf.get_ddPhi(s)    # shape: [len(s), N_via+3] 获取二阶导数基函数

    # 实际任务指定节点位置，计算轨迹
    # p_via = np.array([[0.5, 0.2], [0.5, 0.7], [1.0, 1.0]])  # shape: [2, 3] 指定节点位置
    p_via = np.random.uniform(-1, 1, (N_via, 2))  # shape: [2, N_via] 随机生成节点位置
    p_0 = np.array([0.0, 0.0])
    w = np.concatenate((p_0, p_via.flatten(), np.zeros(4)))
    q = phi @ w
    q = q.reshape(-1, 2)

    # visualize the trajectory
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.plot(q[:, 0], q[:, 1], label='Trajectory', linewidth=4, color='k')
    ax.scatter(p_via[:, 0], p_via[:, 1], color='red', marker='.', label='Via Points', s=300)
    # plot the start point
    ax.scatter(p_0[0], p_0[1], color='blue', marker='*', label='Start Point', s=300)
    ax.set_title('2D Trajectory')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

    # observe the velocity and acceleration
    dq = dphi @ w    # 计算速度，所有evaluation point的速度
    ddq = ddphi @ w  # 计算加速度，所有evaluation point的加速度
    
    dq = dq.reshape(-1, 2)
    ddq = ddq.reshape(-1, 2)

    # 可视化速度和加速度
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # 速度的via points，用竖线表示（因为本身只有eval points的速度，没有via points的速度）
    ax[0].plot(s, dq[:, 0], label='X Position', color='blue', linewidth=2)
    ax[0].plot(s, dq[:, 1], label='Y Position', color='green', linewidth=2)
    ax[0].set_title('Trajectory X and Y Positions Over Time')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(s, ddq[:, 0], label='X Acceleration', color='blue', linewidth=2)
    ax[1].plot(s, ddq[:, 1], label='Y Acceleration', color='green', linewidth=2)
    ax[1].set_title('Acceleration Over Time')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Acceleration')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    main()