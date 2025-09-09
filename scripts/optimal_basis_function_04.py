import sys
import numpy as np
import matplotlib.pyplot as plt
from config import PROJECT_DIR
sys.path.append(PROJECT_DIR)

from planners.obf import OBF

def main():
    # ---------- 基础设置 ----------
    N_via = 2
    N_eval = 100
    T_list = [2.0, 1.0, 0.5]              # 物理总时长（秒）
    h = np.full(N_via, 1.0 / N_via)       # 相位段长，总和=1（相位 s∈[0,1]）
    s = np.linspace(0.0, 1.0, N_eval)        # 相位采样

    # 相位域：一次性构建基函数（与 T 无关）
    obf = OBF(ndof=2)
    obf.setup_task(h)
    Phi_s   = obf.get_Phi(s)
    dPhi_s  = obf.get_dPhi(s)             # 对相位 s 的一阶导
    ddPhi_s = obf.get_ddPhi(s)            # 对相位 s 的二阶导

    # 位置节点与边界速度（物理量）
    p0 = np.array([0.0, 0.0])
    p  = np.array([[0.5, 0.5],
                   [1.0, 1.0]])
    qd0 = np.array([5.0, 0.0])            # 初始物理速度 (x 方向 5, y 方向 0)
    qdT = np.array([0.0, 0.0])

    traj_list, vel_list, acc_list, t_phys_list = [], [], [], []

    for T in T_list:
        # 论文 3.2：将物理边界速度映射到相位导数 ⇒ 乘 T 放进 w
        w = np.concatenate((p0, p.flatten(), T * qd0, T * qdT))

        # 位置、速度、加速度（把相位导数按 1/T、1/T^2 缩回物理量）
        q   = (Phi_s  @ w).reshape(-1, 2)
        dq  = (dPhi_s @ w).reshape(-1, 2) / T
        ddq = (ddPhi_s@ w).reshape(-1, 2) / T**2

        traj_list.append(q)
        vel_list.append(dq)
        acc_list.append(ddq)
        t_phys_list.append(s * T)         # 物理时间轴（0→T）

    # ---------- 画路径 ----------
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(p[:, 0], p[:, 1], color='red', s=80, zorder=3, label='Via points')
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, T in enumerate(T_list):
        ax.plot(traj_list[i][:, 0], traj_list[i][:, 1],
                linewidth=3, color=colors[i], label=f'T={T}')
    ax.set_title('Optimal path (phase modeling)')
    ax.set_xlabel('$x_q$'); ax.set_ylabel('$y_q$')
    ax.grid(alpha=0.3); ax.legend()
    plt.show()

    # ---------- 画 x 方向速度 ----------
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, T in enumerate(T_list):
        t = t_phys_list[i]
        ax.plot(t, vel_list[i][:, 0], linewidth=3, color=colors[i], label=f'T={T}')
        ax.scatter([t[0], t[-1]], [vel_list[i][0, 0], vel_list[i][-1, 0]],
                   color='red', s=40, zorder=3)
        ax.plot([t[0], t[-1]], [vel_list[i][0, 0], vel_list[i][-1, 0]],
                color='red', linestyle='--', linewidth=1.5)
    ax.set_title('Velocity in $x_q$ (phase→time scaling)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('$\dot x_q$')
    ax.grid(alpha=0.3); ax.legend()
    plt.show()

    # ---------- 画 x 方向加速度 ----------
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, T in enumerate(T_list):
        t = t_phys_list[i]
        ax.plot(t, acc_list[i][:, 0], linewidth=3, color=colors[i], label=f'T={T}')
        ax.scatter([t[0], t[-1]], [acc_list[i][0, 0], acc_list[i][-1, 0]],
                   color='red', s=40, zorder=3)
        ax.plot([t[0], t[-1]], [acc_list[i][0, 0], acc_list[i][-1, 0]],
                color='red', linestyle='--', linewidth=1.5)
    ax.set_title('Acceleration in $x_q$ (phase→time scaling)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('$\ddot x_q$')
    ax.grid(alpha=0.3); ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
