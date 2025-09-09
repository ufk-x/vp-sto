import sys
import numpy as np
import matplotlib.pyplot as plt
from config import PROJECT_DIR
sys.path.append(PROJECT_DIR)

from planners.obf import OBF

def main():
    N_via = 3
    T_list = [2.0, 1.0, 0.5] # 设置一个总的时间进行缩放，unit = [s]
    # 均匀分布T_list，总的是1
    # T_list = [1.0/N_via]*N_via
    h = np.full(N_via, 1.0 / N_via)  # 默认相位是[0, 1]， 无量纲，via points是均匀分布
    # create a figure with 3 subfigures
    phi_list = []
    dphi_list = []
    ddphi_list = []
    for i, T in enumerate(T_list):
        obf = OBF(ndof=2)
        obf.setup_task(h*T)  # h*T，每段都是真实时间
        t = np.linspace(0, np.sum(h)*T, 100)  # np.sum(h)*T 是总的时间
        phi = obf.get_Phi(t)
        dphi = obf.get_dPhi(t)
        ddphi = obf.get_ddPhi(t)
        phi_list.append(phi)
        dphi_list.append(dphi)
        ddphi_list.append(ddphi)

    p = np.array([[0.5, 0.2], [0.5, 0.5], [1.0, 1.0]])
    p_0 = np.array([0.0, 0.0])
    dp0 = np.array([5.0, 0.0])

    traj_list = []
    vel_list = []
    acc_list = []
    for i in range(len(T_list)):
        w = np.concatenate((p_0, p.flatten(), dp0, np.zeros(2)))
        traj_i = phi_list[i] @ w
        traj_i = traj_i.reshape(-1, 2)
        vel_i = dphi_list[i] @ w
        vel_i = vel_i.reshape(-1, 2)
        acc_i = ddphi_list[i] @ w
        acc_i = acc_i.reshape(-1, 2)
        traj_list.append(traj_i)
        vel_list.append(vel_i)
        acc_list.append(acc_i)

    # visualization
    # trajectory
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(p[:, 0], p[:, 1], color='red', marker='.', label='Via Points', s=300)
    for i in range(len(T_list)):
        ax.plot(traj_list[i][:, 0], traj_list[i][:, 1], label=f'Trajectory {i+1}, , Time {T_list[i]}', linewidth=4)
        ax.set_title('2D Trajectory')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.show()


    # visualize the trajectory along x
    fig, ax = plt.subplots(figsize=(14, 14))
    for i in range(len(T_list)):
        t = np.linspace(0, T_list[i], 100)  # 每个时间段的时间
        ax.plot(t, vel_list[i][:, 0], label=f'Vel {i+1}, Time {T_list[i]}', linewidth=4)
        # 起点用红色突出
        ax.scatter(t[0], vel_list[i][0, 0], color='red', marker='o', s=100)
        ax.scatter(t[-1], vel_list[i][-1, 0], color='red', marker='o', s=100)
        # 画一条直线，连接起点和终点
        ax.plot([t[0], t[-1]], [vel_list[i][0, 0], vel_list[i][-1, 0]], color='red', linestyle='--', linewidth=2)
        ax.set_title('2D Vel')
        ax.set_xlabel('t')
        ax.set_ylabel('q_x')
        ax.grid(True, alpha=0.3)
        ax.legend()        
    plt.show()

    # acceleration
    fig, ax = plt.subplots(figsize=(14, 14))
    for i in range(len(T_list)):
        t = np.linspace(0, T_list[i], 100)
        ax.plot(t, acc_list[i][:, 0], label=f'Acc {i+1}, Time {T_list[i]}', linewidth=4)
        # 起点用红色突出
        ax.scatter(t[0], acc_list[i][0, 0], color='red', marker='o', s=100)
        ax.scatter(t[-1], acc_list[i][-1, 0], color='red', marker='o', s=100)
        # 画一条直线，连接起点和终点
        ax.plot([t[0], t[-1]], [acc_list[i][0, 0], acc_list[i][-1, 0]], color='red', linestyle='--', linewidth=2)
        ax.set_title('2D Acc')
        ax.set_xlabel('t')
        ax.set_ylabel('q_x')
        ax.grid(True, alpha=0.3)
        ax.legend()        
    plt.show()


if __name__ == "__main__":
    main()