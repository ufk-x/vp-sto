import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
import cma

# 选择目标函数: 'rosenbrock', 'rastrigin', 'ackley', 'himmelblau', 'griewank', 'schwefel'
FUNC_NAME = 'himmelblau'   # ← 换这个名字即可

def obj_and_plot_setup(name):
    name = name.lower()
    if name == 'rosenbrock':
        # 全局最优 (1,1) -> 0
        def f(x): return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
        xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0
        minima = [(1.0, 1.0)]
    elif name == 'rastrigin':
        # 全局最优 (0,0) -> 0
        def f(x): return 20 + (x[0]**2 - 10*np.cos(2*np.pi*x[0])) + (x[1]**2 - 10*np.cos(2*np.pi*x[1]))
        xmin, xmax, ymin, ymax = -5.5, 5.5, -5.5, 5.5
        minima = [(0.0, 0.0)]
    elif name == 'ackley':
        # 全局最优 (0,0) -> 0
        def f(x):
            a, b, c = 20, 0.2, 2*np.pi
            s1 = 0.5*(x[0]**2 + x[1]**2)
            s2 = 0.5*(np.cos(c*x[0]) + np.cos(c*x[1]))
            return -a*np.exp(-b*np.sqrt(2*s1)) - np.exp(s2) + a + np.e
        xmin, xmax, ymin, ymax = -5.0, 5.0, -5.0, 5.0
        minima = [(0.0, 0.0)]
    elif name == 'himmelblau':
        # 四个全局最优 -> 0
        def f(x):
            X, Y = x[0], x[1]
            return (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
        xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
        minima = [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
    elif name == 'griewank':
        # 全局最优 (0,0) -> 0
        def f(x):
            return (x[0]**2 + x[1]**2)/4000.0 - np.cos(x[0]/np.sqrt(1)) * np.cos(x[1]/np.sqrt(2)) + 1
        xmin, xmax, ymin, ymax = -6.0, 6.0, -6.0, 6.0
        minima = [(0.0, 0.0)]
    elif name == 'schwefel':
        # 全局最优 (420.9687..., 420.9687...) -> ~0 在 2D
        def f(x):
            return 418.9829*2 - (x[0]*np.sin(np.sqrt(abs(x[0]))) + x[1]*np.sin(np.sqrt(abs(x[1]))))
        xmin, xmax, ymin, ymax = -500.0, 500.0, -500.0, 500.0
        minima = [(420.968746, 420.968746)]
    else:
        raise ValueError("Unknown function name.")
    return f, (xmin, xmax, ymin, ymax), minima

def covariance_ellipse(cov, mean, nstd=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width, height = 2 * nstd * np.sqrt(np.clip(vals, 1e-12, None))
    return Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, lw=2)

# 目标函数与绘图设置
f, (xmin, xmax, ymin, ymax), minima = obj_and_plot_setup(FUNC_NAME)

# CMA-ES 设置
x0 = [2.0, 1.5]     # 可以改得远一些看路径
sigma0 = 1.5
opts = dict(seed=42, popsize=80, maxiter=40, verb_log=0, verb_disp=0)
es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

# 运行并记录历史
history = []
while not es.stop():
    xs = es.ask()
    vals = [f(x) for x in xs]
    es.tell(xs, vals)
    mean = np.array(es.mean)
    sigma = es.sigma
    C = es.C
    cov = (sigma**2) * C
    history.append({"xs": np.array(xs), "vals": np.array(vals), "mean": mean.copy(), "cov": cov.copy()})
    if len(history) >= opts["maxiter"]:
        break

print("CMA-ES result:", es.result)

# 画等高线
res = 300
xg = np.linspace(xmin, xmax, res)
yg = np.linspace(ymin, ymax, res)
X, Y = np.meshgrid(xg, yg)
Z = np.zeros_like(X)
for i in range(res):
    for j in range(res):
        Z[i, j] = f([X[i, j], Y[i, j]])

fig, ax = plt.subplots(figsize=(7, 7))
levels = 30
ax.contour(X, Y, Z, levels=levels, linewidths=0.8)
ax.set_aspect('equal', 'box')
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_title(f"CMA-ES on {FUNC_NAME.capitalize()}")
ax.set_xlabel("x"); ax.set_ylabel("y")

# 画已知全局最优点（若有）
for (mx, my) in minima:
    ax.plot(mx, my, 'g*', markersize=10, label='global min' if 'global min' not in [l.get_label() for l in ax.lines] else "_nolegend_")

samples_scatter = ax.scatter([], [], s=18, alpha=0.5)
mean_pt, = ax.plot([], [], marker='x', markersize=8, linestyle='None', mew=2, color='r')
best_pt, = ax.plot([], [], marker='o', markersize=6, linestyle='None', color='orange')
ellipse_artist = None
title = ax.set_title("")

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
    xs = state["xs"]; vals = state["vals"]
    mean = state["mean"]; cov = state["cov"]
    best_idx = np.argmin(vals); best = xs[best_idx]; best_val = vals[best_idx]

    samples_scatter.set_offsets(xs)
    mean_pt.set_data([mean[0]], [mean[1]])
    best_pt.set_data([best[0]], [best[1]])

    if ellipse_artist is not None:
        ellipse_artist.remove()
    ellipse_artist = covariance_ellipse(cov, mean, nstd=2.0)
    ax.add_patch(ellipse_artist)

    title.set_text(f"{FUNC_NAME.capitalize()}  |  Iter {frame+1}/{len(history)}  |  best f ≈ {best_val:.4e}  |  mean=({mean[0]:.2f}, {mean[1]:.2f})")
    return samples_scatter, mean_pt, best_pt, ellipse_artist, title

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=len(history), interval=400, blit=True)

SAVE_ANIMATION = False
if not SAVE_ANIMATION:
    plt.show()
    exit(0)
    
# 保存动画（优先 mp4, 其次 gif）
out_mp4 = f"cma_es_{FUNC_NAME}.mp4"
out_gif = f"cma_es_{FUNC_NAME}.gif"
saved = None
try:
    Writer = animation.FFMpegWriter
    writer = Writer(fps=3, metadata=dict(artist="CMA-ES demo"), bitrate=2000)
    anim.save(out_mp4, writer=writer)
    saved = out_mp4
except Exception:
    try:
        from matplotlib.animation import PillowWriter
        anim.save(out_gif, writer=PillowWriter(fps=3))
        saved = out_gif
    except Exception:
        pass

print("Animation saved to:", saved or "(display only)")
plt.show()  # 如需交互查看
