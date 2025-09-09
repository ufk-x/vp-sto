import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
import cma

# =========================
# 1) 目标函数与基础设置
# =========================
def f(x):
    # 最小化 x^2 + y^2
    return sum(xi**2 for xi in x)

x0 = [0, 0]     # 初始猜测点
sigma0 = 1.0    # 初始方差（step-size）

# 为了演示更清晰，这里把起点改远一点（可选）
x0 = [2.0, 1.5]

# CMA-ES 选项
opts = {
    "seed": 42,
    "popsize": 60,      # 每次迭代样本数量
    "maxiter": 35,      # 迭代次数上限（演示用）
    "verb_log": 0,
    "verb_disp": 0
}

# =========================
# 2) 运行 CMA-ES（ask/tell 回路，记录历史）
# =========================
es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

history = []  # 记录每次迭代的：样本、均值、协方差

while not es.stop():
    xs = es.ask()                     # 采样一批候选解（shape: popsize x dim）
    vals = [f(x) for x in xs]         # 评估
    es.tell(xs, vals)                 # 反馈给优化器
    mean = np.array(es.mean)          # 当前均值
    sigma = es.sigma                  # 当前全局步长（标量）
    C = es.C                          # 当前协方差的“形状矩阵”（对称正定）
    cov = (sigma**2) * C              # 实际协方差矩阵 Σ = σ^2 C

    # 记录
    history.append({
        "xs": np.array(xs),
        "vals": np.array(vals),
        "mean": mean.copy(),
        "cov": cov.copy()
    })

    # 控制迭代次数用于动画演示（可删）
    if len(history) >= opts["maxiter"]:
        break

print("CMA-ES result:", es.result)

# =========================
# 3) 绘图与动画
# =========================
def covariance_ellipse(cov, mean, nstd=2.0):
    """根据 2x2 协方差画 nstd-σ 椭圆。"""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # 主轴方向角
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # 半轴长度：nstd * sqrt(eigenvalues)
    width, height = 2 * nstd * np.sqrt(vals)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, lw=2)

# 目标函数等高线
xg = np.linspace(-3.0, 3.0, 200)
yg = np.linspace(-3.0, 3.0, 200)
X, Y = np.meshgrid(xg, yg)
Z = X**2 + Y**2

fig, ax = plt.subplots(figsize=(6, 6))
cs = ax.contour(X, Y, Z, levels=12, linewidths=1.0)
ax.set_aspect('equal', 'box')
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
ax.set_xlabel("x"); ax.set_ylabel("y")
title = ax.set_title("")

samples_scatter = ax.scatter([], [], s=20, alpha=0.6)
mean_pt, = ax.plot([], [], marker='x', markersize=8, linestyle='None', mew=2)
best_pt, = ax.plot([], [], marker='o', markersize=6, linestyle='None')
ellipse_artist = None

def init():
    samples_scatter.set_offsets(np.empty((0, 2)))
    mean_pt.set_data([], [])
    best_pt.set_data([], [])
    global ellipse_artist
    if ellipse_artist is not None:
        ellipse_artist.remove()
    return samples_scatter, mean_pt, best_pt

def update(frame):
    global ellipse_artist
    state = history[frame]
    xs = state["xs"]
    mean = state["mean"]
    cov = state["cov"]
    vals = state["vals"]
    best_idx = np.argmin(vals)
    best = xs[best_idx]
    best_val = vals[best_idx]

    samples_scatter.set_offsets(xs)
    mean_pt.set_data([mean[0]], [mean[1]])
    best_pt.set_data([best[0]], [best[1]])

    if ellipse_artist is not None:
        ellipse_artist.remove()
    ellipse_artist = covariance_ellipse(cov, mean, nstd=2.0)
    ax.add_patch(ellipse_artist)

    title.set_text(f"Iteration {frame+1}/{len(history)}   best f ≈ {best_val:.4f}   mean = ({mean[0]:.2f}, {mean[1]:.2f})")
    return samples_scatter, mean_pt, best_pt, ellipse_artist, title

anim = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=len(history), interval=400, blit=True
)

SAVE_ANIMATION = False
if not SAVE_ANIMATION:
    plt.show()
    exit(0)

# 保存动画（优先 mp4，失败则 gif）
out_mp4 = "cma_es_convergence.mp4"
out_gif = "cma_es_convergence.gif"
saved = None
try:
    Writer = animation.FFMpegWriter
    writer = Writer(fps=3, metadata=dict(artist="CMA-ES demo"), bitrate=1800)
    anim.save(out_mp4, writer=writer)
    saved = out_mp4
except Exception as e:
    try:
        from matplotlib.animation import PillowWriter
        anim.save(out_gif, writer=PillowWriter(fps=3))
        saved = out_gif
    except Exception as e2:
        pass

print("Animation saved to:", saved)
# 如果你想直接在交互式窗口看动画，可以：plt.show()
