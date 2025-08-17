# VP-STO 项目入门到掌握指南

本文档文件名（可点击）：[Instrument.md](./Instrument.md)

---

## 1. 一句话概述

VP-STO（Via-Point Stochastic Trajectory Optimization）是一个基于 via-point 的随机、无需梯度的轨迹优化器。它用时间连续的 OBF（基函数）表达整条轨迹，使用 CMA-ES 在低维的 via-point 空间搜索，同时内置速度/加速度/边界条件约束，适合离线规划，也可扩展到 MPC。

## 2. 代码结构与职责

- [`vpsto/vpsto.py`](./vpsto/vpsto.py)
  - `VPSTOOptions`: 算法与约束的可配参数。
  - `VPSTOSolution`: 解的容器，保存最优/均值、日志，可按时间采样轨迹。
  - `VPSTO`: 主优化器，封装 CMA-ES 采样-评估-更新流程。

- [`vpsto/vptraj.py`](./vpsto/vptraj.py)
  - `VPTraj`: 轨迹表达与快速批量计算，负责从 via-point 参数 p 生成 q(t)/dq(t)/ddq(t)，以及计算满足限幅的最小时长 `T_min`。

- [`vpsto/obf.py`](./vpsto/obf.py)
  - `OBF`: 构造时间连续的基函数 Phi/dPhi/ddPhi，满足分段多项式的连续性与端点速度条件。

- [`examples/`](./examples/): 2D 玩具示例（避障/预测采样/MPC）。

- [`README.md`](./README.md): 项目特性、安装与动图展示。

## 3. 安装与依赖

- 必需依赖：`numpy`、`threaded`、`pycma`
- 可选（示例笔记本）：`matplotlib`、`shapely`

安装方式：

1. 依赖安装

   ```bash
   pip install numpy
   pip install threaded
   pip install git+https://github.com/CMA-ES/pycma.git@master
   ```

2. 本地安装

   ```bash
   git clone https://github.com/JuJankowski/vp-sto
   cd vp-sto
   pip install .
   ```

3. conda 环境请使用 conda 安装的 pip。

## 4. 核心思想（面试可讲）

- 轨迹表达：y(t)=Phi(t)@w，w 由“节点位置（含 via-points）+ 起末速度”组成，段内光滑且时间连续。
- 低维搜索：优化变量是 via-point 参数 p（维度约为 ndof×N_via），而不是高维的全时序控制。
- 约束内置：
  - 可指定 q0/dq0、qT/dqT（未指定者并入 p 由优化求解）。
  - 未给定总时长 T 时，自动计算满足速度/加速度限幅的最小时长 `T_min`（时间缩放）。
- 随机优化：CMA-ES 在“白化”的 via-point 空间迭代，适合不可导/不连续代价（如碰撞）。
- 复杂度：对 DoF 线性扩展；批量矩阵乘加速评估。

提示：

- `VPSTO` 与 `VPSTOSolution` 在文件：[`vpsto/vpsto.py`](./vpsto/vpsto.py)
- 轨迹与时长计算在文件：[`vpsto/vptraj.py`](./vpsto/vptraj.py)
- 基函数实现在文件：[`vpsto/obf.py`](./vpsto/obf.py)

## 5. 快速上手 API

- 创建选项：`opt = VPSTOOptions(ndof=k)`（类定义：[`vpsto/vpsto.py`](./vpsto/vpsto.py)）
  - `vel_lim`, `acc_lim`: 元素级速度/加速度上限
  - `N_via`, `N_eval`: via-point 数与代价采样点数
  - `pop_size`, `sigma_init`, `max_iter`, `CMA_diagonal`
  - `multithreading`: 是否多线程评估代价
  - `traj_duration`: 固定总时长（设定后将忽略自动 T_min）

- 优化器：`vpsto = VPSTO(opt)`（类定义：[`vpsto/vpsto.py`](./vpsto/vpsto.py)）

- 自定义代价：
  - 非多线程：loss 接收批量候选字典 `{'pos':(B,T,ndof),'vel','acc','T':(B,)}`，返回长度 B 的向量成本。
  - 多线程：loss 每次接收单个候选 `{'pos':(T,ndof), ... , 'T':float}`，返回标量成本。

- 运行：`solution = vpsto.minimize(...)`（方法定义：[`vpsto/vpsto.py`](./vpsto/vpsto.py)）
  - 至少给 T 或 dqT 之一；`dq0` 未给则默认 0。
  - 未给的边界（qT/dqT）会自动并入 p 一起优化。

- 结果：
  - `solution.T_best`：最优时长（属性定义：[`vpsto/vpsto.py`](./vpsto/vpsto.py)）
  - `solution.get_posvelacc(t_array)`：在给定时间采样 pos/vel/acc（方法定义：[`vpsto/vpsto.py`](./vpsto/vpsto.py)）

## 6. 最小可运行示例

```python
import numpy as np
from vpsto.vpsto import VPSTO, VPSTOOptions

# 1) 定义代价：纯时间最优（越短越好）
def loss_batch(candidates):
    # candidates['T'] 形状为 (B,)
    return candidates['T']

# 2) 配置优化器
ndof = 2
opt = VPSTOOptions(ndof)
opt.N_via = 5
opt.N_eval = 100
opt.pop_size = 25
opt.sigma_init = 0.5
opt.max_iter = 300
opt.vel_lim = 0.5 * np.ones(ndof)  # 元素级速度上限
opt.acc_lim = 2.0 * np.ones(ndof)  # 元素级加速度上限
# opt.multithreading = True  # 若代价很重，切换为多线程并改写 loss

vpsto = VPSTO(opt)

# 3) 设置起末边界（示例：给定 q0 与 qT；dq0 默认 0，dqT 未给则并入 p）
q0 = np.array([0.0, 0.0])
qT = np.array([1.0, 1.0])

# 4) 启动优化（未指定 T，内部会求满足限幅的最小 T）
solution = vpsto.minimize(loss_batch, q0=q0, qT=qT)

# 5) 采样最优轨迹
T_best = solution.T_best
ts = np.linspace(0, T_best, int(T_best * 200) + 1)  # 200 Hz 示例
q, dq, ddq = solution.get_posvelacc(ts)
print('T_best =', T_best, ' q shape =', q.shape)
```

多线程版本（重写 loss 为“单候选-标量”）：

```python
opt.multithreading = True

def loss_single(candidate):
    # candidate['pos'] 形状为 (T_eval, ndof)
    # 示例：在时间最优外，加一点平滑度惩罚
    w_time = 1.0
    w_smooth = 1e-3
    T = candidate['T']
    ddq = candidate['acc']
    smooth = np.sum(ddq**2)
    return w_time * T + w_smooth * smooth

solution = vpsto.minimize(loss_single, q0=q0, qT=qT)
```

## 7. 边界条件与维度（重要）

- qT、dqT 都未知：`dim = ndof * (N_via + 1)`（via-points + 末速度）
- 仅 qT 未知：`dim = ndof * N_via`
- 仅 dqT 未知：`dim = ndof * N_via`
- qT、dqT 都已知：`dim = ndof * (N_via - 1)`

这些情况对应 `vptraj.__setup_basis` 中不同的平滑矩阵及其 Cholesky（白化变换）（文件：[`vpsto/vptraj.py`](./vpsto/vptraj.py)）。

## 8. 调参与性能建议

- 搜索：
  - `pop_size` 越大探索越稳但更慢；`sigma_init` 太小易陷局部，太大难收敛。
  - `CMA_diagonal=True` 得到线性复杂度、速度更快但精细度可能下降。

- 表达：
  - `N_via` 控制容量；过小拟合不足，过大易过拟合且优化更难。
  - `N_eval` 控制代价采样密度；快速调参阶段可低，最终验证应升高避免漏检（如碰撞）。

- 约束：
  - 未给定 T 时，自动求 `T_min` 满足限幅；如任务有节拍要求可直接给定 T（实现：[`vpsto/vptraj.py`](./vpsto/vptraj.py) 的 `get_min_duration`）。

- 并行：
  - 代价函数很重（碰撞/复杂仿真）时再启用多线程；便宜代价下线程开销可能抵消收益（线程封装：[`vpsto/vpsto.py`](./vpsto/vpsto.py) 的 `__loss_multithread`）。

## 9. 常见坑

- 未给 T 且未给 dqT：内部会提醒并设 `dqT=0`，最好提前明确你的意图（逻辑在：[`vpsto/vpsto.py`](./vpsto/vpsto.py) 的 `minimize` 输入检查）。
- 多线程模式仍按“批量 loss”写法：会形状不匹配（多线程 loss 接口见：[`vpsto/vpsto.py`](./vpsto/vpsto.py)）。
- 限幅太紧 + `N_via` 太少：可行空间窄，易停滞；先放宽或增大 `N_via`。
- 代价强不连续 + `sigma_init` 太小：早期探索不足；增大 `sigma_init` 或 `pop_size`。
- `N_eval` 太低：可能漏掉尖峰（碰撞/约束超限），最终请提高验证。

## 10. 高效读源码路线

1) `VPSTO.minimize`：四种 p 的组装逻辑与白化；CMA-ES 循环中如何批量生成与评估候选（所在文件：[`vpsto/vpsto.py`](./vpsto/vpsto.py)）。

2) `VPTraj.get_min_duration`：如何由线性基函数关系解析计算满足 vel/acc 的最小时长（时间缩放）（所在文件：[`vpsto/vptraj.py`](./vpsto/vptraj.py)）。

3) `VPTraj.get_trajectory`：如何把 p 拼成 w、做 T 缩放并批量得到 q/dq/ddq（所在文件：[`vpsto/vptraj.py`](./vpsto/vptraj.py)）。

4) `OBF.__get_base` 与 `OBF.__get_P`：分段多项式、连续性与矩阵构造的关键思路（所在文件：[`vpsto/obf.py`](./vpsto/obf.py)）。

## 11. 运行示例与自测

- 直接打开 [`examples/`](./examples/) 下的笔记本运行，观察 `N_via`、`N_eval`、`pop_size` 对速度与质量的影响。
- 在 2D 上先做“纯时间最优”，再加入简单障碍（圆/多边形距离惩罚）体验不可导代价的优势。
- 最终用更高 `N_eval` 重评估与可视化。

示例链接：

- 避障设末位置（本地）： [`examples/2D_collision_avoidance_set_final_position.ipynb`](./examples/2D_collision_avoidance_set_final_position.ipynb)；（远程）： [GitHub 链接](https://github.com/JuJankowski/vp-sto/blob/dev/examples/2D_collision_avoidance_set_final_position.ipynb)
- 预测采样（本地）： [`examples/2D_predictive_sampling.ipynb`](./examples/2D_predictive_sampling.ipynb)；（远程）： [GitHub 链接](https://github.com/JuJankowski/vp-sto/blob/dev/examples/2D_predictive_sampling.ipynb)

## 12. 面试 Q&A 速查

- 为什么 via-point + OBF？
  - 低维、时间连续、快速线性评估，天然满足连续性；时间缩放下易满足限幅。

- 相比梯度法的优势？
  - 不依赖梯度、对不可导/非凸代价更鲁棒；CMA-ES 的探索性降低局部最小风险。

- 复杂度如何随 DoF 扩展？
  - 主要是线性扩展，核心为矩阵乘和批量评估。

- 如何确保速度/加速度约束？
  - 未给 T 时解析求最小可行 `T_min`；给定 T 则直接评估（必要时在代价中软惩罚违反）。

- 边界条件如何处理？
  - 未指定的 qT/dqT 并入参数 p，通过平滑矩阵与 Cholesky 白化使搜索空间良态化。

- 多线程何时有利？
  - 单条轨迹代价评估很重（复杂碰撞/仿真）时，否则线程开销可能不划算。

- 失败模式与缓解？
  - 收敛慢/卡住：增大 `pop_size`/`sigma_init`，调整 `N_via`/`N_eval`，或用 `CMA_diagonal=True`。

- 如何用于 MPC？
  - 在滚动窗内反复求解，复用 `set_initial_guess` 进行 warm-start，减少迭代；代价加入预测/动态障碍项（接口：[`vpsto/vpsto.py`](./vpsto/vpsto.py) 的 `set_initial_guess`）。

## 13. 进一步扩展

- 代价模板：目标达成、路径长度、平滑度、关节限幅软惩罚、障碍距离场等。
- 加速：将代价向量化或移植到 GPU（若合适）。
- 学习结合：用学习先验生成 `p_init` 或自适应 `sigma_init`。
- 约束增强：在代价里加入更多软约束，或设计安全层过滤无效候选。
