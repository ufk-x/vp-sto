import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import cma

# =========================
# 0) 选择目标函数
# =========================
# 可选: 'rosenbrock', 'rastrigin', 'ackley', 'himmelblau', 'griewank', 'schwefel'
FUNC_NAME = 'himmelblau'   # ← 在这里切换函数

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

# 目标函数与绘图边界
f, (xmin, xmax, ymin, ymax), minima = obj_and_plot_setup(FUNC_NAME)

# =========================
# 1) CMA-ES 参数
# =========================
x0 = [2.5, 1.5]   # 初始点
sigma0 = 1.5
opts = dict(seed=42, popsize=80, maxiter=40, verb_log=0, verb_disp=0)
es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

# =========================
# 2) 运行并记录历史
# =========================
history = []
while not es.stop():
    xs = es.ask()
    vals = [f(x) for x in xs]
    es.tell(xs, vals)
    mean = np.array(es.mean)
    sigma = es.sigma
    C = es.C
    cov = (sigma**2) * C
    zs = np.array([f(x) for x in xs])
    history.append({
        "xs": np.array(xs),
        "vals": np.array(vals),
        "zs": zs,
        "mean": mean.copy(),
        "mean_z": f(mean),
        "cov": cov.copy()
    })
    if len(history) >= opts["maxiter"]:
        break

print("CMA-ES result:", es.result)

# =========================
# 3) 准备 3D 曲面
# =========================
res = 160
xg = np.linspace(xmin, xmax, res)
yg = np.linspace(ymin, ymax, res)
X, Y = np.meshgrid(xg, yg)
Z = np.zeros_like(X)
for i in range(res):
    for j in range(res):
        Z[i, j] = f([X[i, j], Y[i, j]])

zmin = float(np.min(Z))
zmax = float(np.max(Z))

# =========================
# 4) 3D 动画
# =========================
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)

# 全局最优点（仅第一个加 label，避免你的那行嵌套三元表达式）
labeled_once = False
for (mx, my) in minima:
    mz = f([mx, my])
    if not labeled_once:
        ax.scatter([mx], [my], [mz], color='lime', s=60, marker='*', depthshade=False, label='global min')
        labeled_once = True
    else:
        ax.scatter([mx], [my], [mz], color='lime', s=60, marker='*', depthshade=False)

if labeled_once:
    ax.legend(loc='upper right')

# 初始 scatter/mean/track
samples_scatter = ax.scatter([], [], [], s=15, c=[], cmap='plasma', depthshade=True, vmin=zmin, vmax=zmax)
mean_pt = ax.scatter([], [], [], s=50, c='r', marker='x', depthshade=False)
track_line, = ax.plot([], [], [], 'r-', lw=2, alpha=0.9)  # 均值轨迹

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
ax.set_title(f"CMA-ES on {FUNC_NAME.capitalize()} (3D)")

track_x, track_y, track_z = [], [], []

def init():
    samples_scatter._offsets3d = ([], [], [])
    samples_scatter.set_array(np.array([]))
    mean_pt._offsets3d = ([], [], [])
    track_line.set_data_3d([], [], [])
    ax.view_init(elev=35, azim=45)
    return samples_scatter, mean_pt, track_line

def update(frame):
    state = history[frame]
    xs = state["xs"]; zs = state["zs"]
    mean = state["mean"]; mean_z = state["mean_z"]

    Xs = xs[:, 0]; Ys = xs[:, 1]; Zs = zs
    samples_scatter._offsets3d = (Xs, Ys, Zs)
    samples_scatter.set_array(zs)  # 颜色映射按函数值

    mean_pt._offsets3d = ([mean[0]], [mean[1]], [mean_z])

    track_x.append(mean[0]); track_y.append(mean[1]); track_z.append(mean_z)
    track_line.set_data_3d(track_x, track_y, track_z)

    # 小幅旋转视角（可注释掉）
    # ax.view_init(elev=35, azim=45 + 0.8*frame)
    return samples_scatter, mean_pt, track_line

anim = animation.FuncAnimation(fig, update, init_func=init,
                               frames=len(history), interval=500, blit=False)

SAVE_ANIMATION = False
if not SAVE_ANIMATION:
    plt.show()
    exit(0)

# 保存动画（优先 mp4，其次 gif）
out_mp4 = f"cma_es_{FUNC_NAME}_3d.mp4"
out_gif = f"cma_es_{FUNC_NAME}_3d.gif"
saved = None
try:
    Writer = animation.FFMpegWriter
    writer = Writer(fps=3, metadata=dict(artist="CMA-ES 3D demo"), bitrate=2500)
    anim.save(out_mp4, writer=writer)
    saved = out_mp4
except Exception:
    try:
        from matplotlib.animation import PillowWriter
        anim.save(out_gif, writer=PillowWriter(fps=3))
        saved = out_gif
    except Exception:
        pass

print("3D animation saved to:", saved or "(display only)")
# plt.show()  # 如需交互查看
