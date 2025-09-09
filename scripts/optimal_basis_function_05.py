import sys
import numpy as np
import matplotlib.pyplot as plt
from config import PROJECT_DIR
sys.path.append(PROJECT_DIR)

from planners.obf import OBF

def main():
    # ---------- 基础设置 ----------
    N_via = 4
    N_eval = 100
    vel_lim_list = [1.0, 1.5, 2.0]          # 物理速度限制（m/s）
    h = np.full(N_via, 1.0 / N_via)         # 相位段长，总和=1（相位 s∈[0,1]）
    s = np.linspace(0.0, 1.0, N_eval)       # 相位采样
    obf = OBF(ndof=2)
    obf.setup_task(h)
    Phi_s   = obf.get_Phi(s)
    dPhi_s  = obf.get_dPhi(s)               # 对相位 s 的一阶导
    ddPhi_s = obf.get_ddPhi(s)              # 对相位 s 的二阶导
    
    # 定义轨迹节点
    p0 = np.array([0.5, 0.0])               # 起始位置
    via_points = np.array([[0.1, 0.3],      # 中间通径点
                          [0.5, 0.5],
                          [0.9, 0.7],
                          [0.5, 1.0]]) 
    
    qd0 = np.array([1.0, 0.0])              # 初始物理速度 
    qdT = np.array([0.0, 0.0])              # 终端物理速度
    
    traj_list, vel_list, acc_list, t_phys_list, T_list = [], [], [], [], []
    w_q = np.concatenate((p0, via_points.flatten()))
    w_dq = np.concatenate((qd0, qdT))
    w = np.concatenate((w_q, w_dq))

    for i, vel_lim in enumerate(vel_lim_list):
        numerator = dPhi_s[2:, :-4] @ w_q
        denominator = dPhi_s[2:, -4:] @ w_dq

        T_dq = np.maximum(np.max(numerator / (vel_lim - denominator)),
                           np.max(-numerator / (vel_lim + denominator)))
        T_list.append(T_dq)
        print(f"对于速度限制 {vel_lim} m/s，计算得到的最小时间尺度 T = {T_dq:.3f} s")
    
    
    for i, T in enumerate(T_list):
        # 速度缩放到物理量
        w_dq = np.concatenate((T * qd0, T * qdT))
        w = np.concatenate((w_q, w_dq))

        # 位置、速度、加速度（把相位导数按 1/T、1/T^2 缩回物理量）
        q   = (Phi_s  @ w).reshape(-1, 2)
        dq  = (dPhi_s @ w).reshape(-1, 2) / T
        ddq = (ddPhi_s@ w).reshape(-1, 2) / T**2

        traj_list.append(q)
        vel_list.append(dq)
        acc_list.append(ddq)
        t_phys_list.append(s * T)         # 物理时间轴（0→T）
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(p0[0], p0[1], color='red', s=80, zorder=3, label='Start Point')
    ax.scatter(via_points[:, 0], via_points[:, 1], color='blue', s=80, zorder=3, label='Via Points')
    ax.scatter(via_points[-1, 0], via_points[-1, 1], color='green', s=80, zorder=3, label='End Point')
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, T in enumerate(T_list):
        ax.plot(traj_list[i][:, 0], traj_list[i][:, 1],
                linewidth=3, color=colors[i], label=f'T={T:.2f} s')
    ax.set_title('Optimal Path with Phase Modeling')
    ax.set_xlabel('$x_q$'); ax.set_ylabel('$y_q$')
    ax.grid(alpha=0.3); ax.legend()
    plt.show()

    # ---------- 画 x 方向速度 ----------
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, T in enumerate(T_list):
        t = t_phys_list[i]
        ax.plot(t, vel_list[i][:, 0], linewidth=3, color=colors[i], label=f'T={T:.2f} s')
        # 画约束，颜色和上面一致
        ax.plot([t[0], t[-1]], [vel_lim_list[i], vel_lim_list[i]], color=colors[i], linestyle='--', linewidth=1.5)
        ax.plot([t[0], t[-1]], [-vel_lim_list[i], -vel_lim_list[i]], color=colors[i], linestyle='--', linewidth=1.5)

        ax.scatter([t[0], t[-1]], [vel_list[i][0, 0], vel_list[i][-1, 0]],
                   color='red', s=40, zorder=3)
        ax.plot([t[0], t[-1]], [vel_list[i][0, 0], vel_list[i][-1, 0]],
                color='red', linestyle='--', linewidth=1.5)
    ax.set_title('Velocity in $x_q$ (Phase to Time Scaling)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('$\dot x_q$')
    ax.grid(alpha=0.3); ax.legend()
    plt.show()






if __name__ == "__main__":
    main()
