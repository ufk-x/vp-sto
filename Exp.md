<!--
 * @Author: Fang Kai[thissfk@qq.com]
 * @Date: 2025-08
 * @LastEditors: Fang Kai[thissfk@qq.com]
 * @LastEditTime: 2025-08
 * @FilePath: Exp.md
 * @Description: 
 *            If you need more information,
 * please contact Fang Kai[thissfk@qq.com] to get an access.   
 * Copyright (c) 2025 by Fang Kai, All Rights Reserved. 
-->
# 实验步骤与复现指南（Exp.md）

本文目标：

- 复现论文“VP-STO: Via-point-based Stochastic Trajectory Optimization for Reactive Robot Behavior”中的若干图像/效果。
- 对比 VP-STO 与 STOMP（Stochastic Trajectory Optimization for Motion Planning）的差异并总结。

相关资料：

- 本地论文 PDF：[`VP-STO_Via-point-based_Stochastic_Trajectory_Optimization_for_Reactive_Robot_Behavior.pdf`](./VP-STO_Via-point-based_Stochastic_Trajectory_Optimization_for_Reactive_Robot_Behavior.pdf)
- 代码主文件：[`vpsto/vpsto.py`](./vpsto/vpsto.py) · [`vpsto/vptraj.py`](./vpsto/vptraj.py) · [`vpsto/obf.py`](./vpsto/obf.py)
- 示例笔记本：[`examples/`](./examples/) 目录（关键：[`2D_collision_avoidance_set_final_position.ipynb`](./examples/2D_collision_avoidance_set_final_position.ipynb)、[`2D_predictive_sampling.ipynb`](./examples/2D_predictive_sampling.ipynb)）

---

## 0. 环境准备

以下提供两种独立环境方式：推荐使用 Miniconda；若不安装 conda，可使用 Python 自带 venv。

### 0.1 在 Ubuntu 20.04 安装 Miniconda（推荐）

```bash
# 下载最新 Miniconda（Python 3.x）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# 运行安装程序（按提示回车/yes；可选择默认目录；建议 yes 执行 conda init）
bash ~/miniconda.sh

# 如安装过程中未执行 init，可手动：
conda init bash

# 重新加载 shell（或重新打开一个终端）
source ~/.bashrc

# 验证安装
conda --version
```

常见问题：

- 若 conda 命令未找到，确认 `~/.bashrc` 中存在 `conda initialize` 段，并已 `source ~/.bashrc`。

### 0.2 使用 conda 创建并激活独立环境

```bash
# 接受某些频道（如 pkgs/main 或 pkgs/r）的服务条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建名为 vpsto 的环境，指定 Python 版本（例如 3.10）
conda create -n vpsto python=3.10 -y

# 激活环境
conda activate vpsto

# 确认当前 python 与 pip 指向该环境
which python
which pip
python -V
pip -V
```

提示：激活环境后，请使用该环境自带的 pip 安装依赖（与本仓库 README 的说明一致）。

### 0.3 不安装 conda 的替代方案：使用 Python venv

```bash
# 安装 venv 与 pip（若系统尚未安装）
sudo apt update && sudo apt install -y python3-venv python3-pip

# 在项目根目录创建虚拟环境（.venv 目录）
python3 -m venv .venv

# 激活虚拟环境（当前 shell 有效）
source .venv/bin/activate

# 验证版本与路径
python -V
pip -V
```

提示：使用 venv 时，后续所有 pip 安装命令都应在激活后的同一终端执行。

### 0.4 在“已激活”的环境中安装项目依赖

```bash
pip install numpy
pip install threaded
pip install cma
pip install git+https://github.com/CMA-ES/pycma.git@master
pip install matplotlib shapely jupyter imageio
```

### 0.5 本地安装本项目（可选，开发模式建议）

```bash
# 确保当前位于仓库根目录
pip install .
```

### 0.6 可复现性建议（随机种子）

- 在你运行的 Python 或 Notebook 顶部设置随机种子，以稳定采样结果（CMA-ES 也依赖 numpy 随机数）：

```python
import numpy as np
np.random.seed(42)
```

- 如需更严格的随机控制，可在创建优化器前设置：

```python
import cma
cma.CMAOptions()['seed'] = 42  # 若使用原生 cma 接口；本项目中可先设置 np.random.seed
```

---

## 1. 复现论文图像/效果

说明：论文中的核心可视化包含（1）在复杂障碍环境中的避障与时间最优轨迹；（2）用于 MPC 的动态场景导航；（3）轨迹的速度/加速度剖面等。下面给出一步步复现流程与导出图像的方法。

### 1.1 静态障碍环境：避障 + 时间最优（对应论文的避障示意）

- 打开本地笔记本：[`examples/2D_collision_avoidance_set_final_position.ipynb`](./examples/2D_collision_avoidance_set_final_position.ipynb)

- 运行顺序：

  1. 在第一格加入随机种子代码（可选，便于复现一致图像）：

     ```python
     import numpy as np
     np.random.seed(42)
     ```

  1. 逐格运行，观察优化过程/最终轨迹。

  1. 将示例中的参数与 README 动图风格对齐（建议）：
     - `N_via = 5`、`N_eval = 100~200`（最终导出图像可提高到 300）
     - `pop_size = 25~40`、`sigma_init = 0.4~0.6`、`max_iter = 500~1000`
     - 合理的 `vel_lim`/`acc_lim`（例如 2D: vel 0.5、acc 2.0，参考 `Instrument.md`）

  1. 导出关键帧与图像：在生成最终轨迹后，插入如下绘图保存代码：

     ```python
     import matplotlib.pyplot as plt
     import os
     
     # 确保 media 目录存在
     os.makedirs('media', exist_ok=True)
     
     # 保存图像的辅助函数
     def save_fig(path):
         plt.gcf().set_size_inches(6, 6)
         plt.tight_layout()
         plt.savefig(path, dpi=200, bbox_inches='tight')
         print(f"✅ 图像已保存: {path}")
     
     # 绘制静态障碍避障轨迹图
     fig, ax = plt.subplots(figsize=(6, 6))
     
     # 设置坐标范围
     ax.set_xlim([q_min[0], q_max[0]])
     ax.set_ylim([q_min[1], q_max[1]])
     
     # 绘制起点和终点
     ax.scatter([q0[0]], [q0[1]], c='green', s=100, marker='o', label='Start', zorder=5)
     ax.scatter([qd[0]], [qd[1]], c='red', s=100, marker='x', label='Goal', zorder=5)
     
     # 绘制障碍物
     for pol in env.poly_list:
         ax.add_patch(patches.Polygon(pol, facecolor='gray', alpha=0.7, edgecolor='black'))
     
     # 绘制优化后的轨迹（使用 pos 变量而不是 q）
     ax.plot(pos[:,0], pos[:,1], 'r-', lw=3, label='VP-STO Trajectory', zorder=4)
     
     # 设置图像属性
     ax.set_xlabel('X Position')
     ax.set_ylabel('Y Position')
     ax.set_title('VP-STO Static Obstacle Avoidance')
     ax.legend()
     ax.grid(True, alpha=0.3)
     ax.set_aspect('equal')
     
     # 保存图像
     plt.tight_layout()
     plt.savefig('media/exp_static_obstacles_traj.png', dpi=200, bbox_inches='tight')
     print(f"✅ 静态避障轨迹图已保存: media/exp_static_obstacles_traj.png")
     print(f"📈 轨迹持续时间: {sol.T_best:.2f}s")
     plt.show()
     
     # 同时保存速度和加速度剖面图（额外bonus）
     plt.figure(figsize=(10, 6))
     
     # 速度剖面
     plt.subplot(2, 1, 1)
     plt.plot(t_traj, vel[:, 0], 'b-', label='X Velocity')
     plt.plot(t_traj, vel[:, 1], 'r-', label='Y Velocity')
     plt.axhline(y=opt.vel_lim[0], color='b', linestyle='--', alpha=0.5, label='X Vel Limit')
     plt.axhline(y=opt.vel_lim[1], color='r', linestyle='--', alpha=0.5, label='Y Vel Limit')
     plt.axhline(y=-opt.vel_lim[0], color='b', linestyle='--', alpha=0.5)
     plt.axhline(y=-opt.vel_lim[1], color='r', linestyle='--', alpha=0.5)
     plt.ylabel('Velocity (m/s)')
     plt.title('Velocity Profile')
     plt.legend()
     plt.grid(True, alpha=0.3)
     
     # 加速度剖面
     plt.subplot(2, 1, 2)
     plt.plot(t_traj, acc[:, 0], 'b-', label='X Acceleration')
     plt.plot(t_traj, acc[:, 1], 'r-', label='Y Acceleration')
     plt.axhline(y=opt.acc_lim[0], color='b', linestyle='--', alpha=0.5, label='X Acc Limit')
     plt.axhline(y=opt.acc_lim[1], color='r', linestyle='--', alpha=0.5, label='Y Acc Limit')
     plt.axhline(y=-opt.acc_lim[0], color='b', linestyle='--', alpha=0.5)
     plt.axhline(y=-opt.acc_lim[1], color='r', linestyle='--', alpha=0.5)
     plt.xlabel('Time (s)')
     plt.ylabel('Acceleration (m/s²)')
     plt.title('Acceleration Profile')
     plt.legend()
     plt.grid(True, alpha=0.3)
     
     plt.tight_layout()
     plt.savefig('media/exp_profiles_q_dq_ddq.png', dpi=200, bbox_inches='tight')
     print(f"✅ 速度/加速度剖面图已保存: media/exp_profiles_q_dq_ddq.png")
     plt.show()
     ```

- 期望产出：`media/exp_static_obstacles_traj.png`（类似论文中静态场景的避障轨迹图）。

### 1.2 动态场景：预测采样 / MPC 导航（对应论文的动态环境动画）

- 打开本地笔记本：[`examples/2D_predictive_sampling.ipynb`](./examples/2D_predictive_sampling.ipynb)

- 运行顺序：

  1. 在第一格加入随机种子（可选）：

     ```python
     import numpy as np
     np.random.seed(123)
     ```

  1. 逐格运行，并确保示例中包含对移动障碍/预测的建模。

  1. 将帧导出为图片（或 GIF）：

     ```python
     import matplotlib.pyplot as plt
     import imageio.v2 as imageio
   
     frames = []
     for k, qk in enumerate(trajectory_seq):  # trajectory_seq 为每个 MPC 步的轨迹/状态，可参考示例变量
         plt.figure()
         # ... 绘制动态障碍与当前规划轨迹 qk ...
         plt.axis('equal')
         plt.tight_layout()
         frame_path = f'media/exp_mpc_frame_{k:03d}.png'
         plt.savefig(frame_path, dpi=150)
         plt.close()
         frames.append(imageio.imread(frame_path))
   
     imageio.mimsave('media/exp_mpc_navigation.gif', frames, fps=10)
     ```

- 期望产出：`media/exp_mpc_navigation.gif`（与论文/README 中动图风格一致）。

提示：仓库已有示例动图 [`media/mpc_animation.gif`](./media/mpc_animation.gif)，你也可以直接对比自制 GIF 与其视觉效果是否一致。

### 1.3 轨迹剖面：位姿/速度/加速度曲线（与论文曲线类图一致）

- 使用 `VPSTOSolution.get_posvelacc(t_array)` 对最优解采样，绘制 q/dq/ddq 随时间变化曲线：

```python
import numpy as np
import matplotlib.pyplot as plt
from vpsto.vpsto import VPSTO, VPSTOOptions

# ... 你的优化流程，得到 solution ...
T = solution.T_best
ts = np.linspace(0, T, int(T * 200) + 1)
q, dq, ddq = solution.get_posvelacc(ts)

labels = ['q', 'dq', 'ddq']
series = [q, dq, ddq]

plt.figure(figsize=(10, 6))
for i, (name, data) in enumerate(zip(labels, series)):
    plt.subplot(3, 1, i+1)
    plt.plot(ts, data)
    plt.title(name)
    plt.xlabel('t [s]')
    plt.ylabel(name)
    plt.grid(True)
plt.tight_layout()
plt.savefig('media/exp_profiles_q_dq_ddq.png', dpi=200)
plt.show()
```

- 期望产出：`media/exp_profiles_q_dq_ddq.png`（不同自由度的曲线叠加在同一子图内）。

---

## 2. VP-STO 与 STOMP 的对比与总结

背景：

- VP-STO（本项目）：基于 via-point 的时间连续轨迹表达 + CMA-ES 随机优化；内置边界与速/加约束处理，支持自动最小时长 `T_min`。
- STOMP（Kalakrishnan et al., 2011）：基于随机滚动（noisy rollouts）的轨迹优化方法，常用于运动规划（MoveIt 等生态中也较常见）。

对比要点：

1. 轨迹参数化

- VP-STO：via-point + OBF 连续基函数，搜索低维 via-point 参数 p；整条轨迹由线性基组合得到。
- STOMP：通常在固定时间离散栅格上优化整个轨迹（每个时间步的状态/控制），维度与离散步数成正比。

1. 优化机制与“梯度”使用

- VP-STO：使用 CMA-ES 全局式随机优化，不依赖解析梯度；对不可导/不连续代价（如碰撞惩罚）更鲁棒。
- STOMP：通过对噪声滚动的成本评估，形成期望改进方向（本质上是采样估计的“梯度样”信息），再对整条轨迹进行更新与平滑。

1. 约束与时间处理

- VP-STO：
  - 可显式给定 q0/dq0、qT/dqT；未给定的边界合入参数一并优化。
  - 内置速度/加速度限幅：未给 T 时自动求最小可行 `T_min`，避免显式硬约束求解。
- STOMP：
  - 多以软约束/代价项形式体现在更新中；
  - 时间通常固定为均匀离散，时间最优需额外设计（非内置）。

1. 复杂度与扩展性

- VP-STO：维度主要由 `ndof × N_via` 决定，DoF 线性扩展；评估通过大矩阵乘可高效批量化。
- STOMP：维度与离散步数 × DoF 相关；每轮需要多条噪声滚动的评估与加权更新。

1. 局部极小与探索

- VP-STO：CMA-ES 具备较强的全局探索能力，较易跳出局部极小；但也依赖种群规模与步长设置。
- STOMP：通过注入噪声和期望改进可一定程度避免局部极小，但对初始化与超参数敏感。

1. 超参数与调参经验

- VP-STO：`N_via`、`N_eval`、`pop_size`、`sigma_init`、`max_iter`、`CMA_diagonal`、限幅等。
- STOMP：滚动条数（samples）、噪声方差、平滑/正则系数、步长、迭代次数、代价项权重等。

1. 适用场景建议

- 若问题强调时间连续表达、需要内置速/加约束、且代价不可导/包含碰撞：VP-STO 更便利与稳健。
- 若已有成熟 STOMP 集成（如 MoveIt pipeline），或任务以离散时间网格的路径细化为主：STOMP 更易融入现有系统。

简要结论：

- 二者都是采样驱动、对不可导代价友好的优化框架；
- VP-STO 在“低维参数化 + 内置限幅与时间缩放”上更简洁，适合时间最优与复杂约束的快速原型；
- STOMP 在“已有生态集成、离散网格规划”中实用，但时间最优与硬约束需额外设计/调参。

参考与链接：

- 本项目与代码：[`vpsto/vpsto.py`](./vpsto/vpsto.py) · [`vpsto/vptraj.py`](./vpsto/vptraj.py) · [`vpsto/obf.py`](./vpsto/obf.py)
- 论文 PDF（本地）：[`VP-STO_Via-point-based_Stochastic_Trajectory_Optimization_for_Reactive_Robot_Behavior.pdf`](./VP-STO_Via-point-based_Stochastic_Trajectory_Optimization_for_Reactive_Robot_Behavior.pdf)
- STOMP 原始论文：Kalakrishnan et al., "STOMP: Stochastic Trajectory Optimization for Motion Planning", 2011（可在公开学术资源中获取）。

---

## 3. 复现实验清单（Checklist）

- [ ] 配置依赖与环境（0 节）。
- [ ] 运行静态障碍避障示例，导出 `media/exp_static_obstacles_traj.png`（1.1 节）。
- [ ] 运行预测采样/MPC 示例，生成 `media/exp_mpc_navigation.gif`（1.2 节）。
- [ ] 绘制 q/dq/ddq 剖面，导出 `media/exp_profiles_q_dq_ddq.png`（1.3 节）。
- [ ] 对比要点阅读并结合任务选择方法（2 节）。
