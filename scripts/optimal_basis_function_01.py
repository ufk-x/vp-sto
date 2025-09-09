"""
这个脚本的目的是展示如何使用最优基函数 (Optimal Basis Function, OBF) 来生成轨迹。

基函数本身只取决于节点数和维度数，与具体的节点位置无关。
因此基函数可以预先计算并存储，以便在需要时快速调用。

在这个例子中，我们将展示如何生成一维轨迹，并可视化基函数、轨迹、速度和加速度。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from config import PROJECT_DIR
sys.path.append(PROJECT_DIR)

from planners.obf import OBF

def main(figsize=(18, 12)):
    N_via = 2
    N_eval = 100
    h = np.full(N_via, 1.0 / N_via)  # shape: [N_via] 每个节点之间的时间间隔
    obf = OBF(ndof=1)
    obf.setup_task(h)
    print(np.sum(h))
    s = np.linspace(0, np.sum(h), N_eval)  # 考虑相位
    phi = obf.get_Phi(s)        # shape: [len(s), N_via+3] 获取基函数
    dphi = obf.get_dPhi(s)      # shape: [len(s), N_via+3] 获取一阶导数基函数
    ddphi = obf.get_ddPhi(s)    # shape: [len(s), N_via+3] 获取二阶导数基函数

    # 可视化基函数，任务确定以后，基函数就固定了，可以预先计算并存储，以便后续快速调用
    fig, ax = plt.subplots(figsize = figsize)
    for i in range(phi[:,:-2].shape[1]):
        ax.plot(s, phi[:, i], label=f'Phi_{i}(t)', linewidth=2)
    for i in range(phi[:,-2:].shape[1]):
        ax.plot(s, phi[:, -i-1], label=f'dPhi_{i}(t)', linestyle='--', linewidth=2)
    ax.set_title('Basis Function Phi(t)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phi')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

    # 实际任务指定节点位置，计算轨迹
    # p_via = np.array([[0.3, 0.8, -0.6, 0.6]])   # shape: [1, 4] 指定节点位置
    p_via = np.random.uniform(-1, 1, (1, N_via))  # shape: [1, N_via] 随机生成节点位置
    p_0 = np.array([0])
    w = np.concatenate((p_0, p_via.flatten(), np.zeros(2)))  # 包含边界条件的权重
    print(w.shape)
    q = phi @ w 

    # plot traj
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s, q, label='Trajectory', linewidth=2)
    ax.scatter(np.concatenate(([0], np.cumsum(h))), np.concatenate((p_0, p_via.flatten())), 
               color='red', marker='.', label='Via Points', s=300)
    ax.set_title('Trajectory with Via Points')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, np.sum(h)+0.1)
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.show()

    # observe the velocity and acceleration
    dq = dphi @ w    # 计算速度，所有evaluation point的速度
    ddq = ddphi @ w  # 计算加速度，所有evaluation point的加速度
    
    # 可视化速度和加速度
    fig, ax = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ax[0].plot(s, dq, label='Velocity', color='orange', linewidth=2)
    # 速度的via points，用竖线表示（因为本身只有eval points的速度，没有via points的速度）
    for t in np.cumsum(h):
        ax[0].axvline(t, color='red', linestyle='--', alpha=0.5)
    ax[0].set_title('Velocity')
    ax[0].set_ylabel('Velocity')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()  

    # 可视化加速度
    ax[1].plot(s, ddq, label='Acceleration', color='green', linewidth=2)
    for t in np.cumsum(h):
        ax[1].axvline(t, color='red', linestyle='--', alpha=0.5)
    ax[1].set_title('Acceleration')
    ax[1].set_ylabel('Acceleration')
    ax[1].grid(True, alpha=0.3)     
    ax[1].legend()

    ax[2].plot(s, q, label='Trajectory', color='blue', linewidth=2)
    for t in np.cumsum(h):
        ax[2].axvline(t, color='red', linestyle='--', alpha=0.5)
    ax[2].set_title('Trajectory')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Position')    
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()
    ax[2].scatter(np.concatenate(([0], np.cumsum(h))), np.concatenate((p_0, p_via.flatten())), 
            color='red', marker='.', label='Via Points', s=300)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()