import os
import sys
import numpy as np

from config import PLANNER_PATH
sys.path.append(PLANNER_PATH)

from planners.vpsto import VPSTO, VPSTOOptions
from planners.vptraj import VPTraj
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, MultiPolygon, LineString
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import datetime

class CollisionEnvironment():
    """
    多边形障碍物环境类
    
    使用Shapely库创建复杂的2D多边形障碍物环境，
    支持精确的轨迹-障碍物交集计算。
    """
    
    def __init__(self):
        """初始化多边形障碍物环境"""
        self.poly_list = []
        
        # 位置在工作空间的左下区域
        self.poly_list.append(np.array([
            [0.1, 0.13],   # 顶点1
            [0.23, 0.12],  # 顶点2  
            [0.19, 0.28],  # 顶点3
            [0.1, 0.32],   # 顶点4
            [0.16, 0.2]    # 顶点5
        ]))
        
        # 位置在工作空间的中上区域
        self.poly_list.append(np.array([
            [0.25, 0.34],  # 顶点1
            [0.31, 0.35],  # 顶点2
            [0.32, 0.41],  # 顶点3
            [0.27, 0.44],  # 顶点4
            [0.23, 0.4]    # 顶点5
        ]))
        
        # 位置在工作空间的右下区域
        self.poly_list.append(np.array([
            [0.35, 0.12],  # 顶点1
            [0.38, 0.1],   # 顶点2
            [0.41, 0.11],  # 顶点3
            [0.42, 0.21],  # 顶点4
            [0.35, 0.24]   # 顶点5
        ]))
        
        # 使用Shapely创建多边形联合体，便于高效的几何计算
        self.multi_poly = MultiPolygon([
            Polygon(self.poly_list[0]), 
            Polygon(self.poly_list[1]), 
            Polygon(self.poly_list[2])
        ])
        
    def getTrajDist(self, pts):
        """
        计算轨迹与障碍物的交集长度
        
        参数:
        - pts: (N, 2)数组，轨迹上的点序列
        
        返回:
        - 轨迹与所有障碍物交集的总长度
        
        注意：
        - 返回值>0表示发生碰撞
        - 返回值越大表示碰撞越严重
        """
        # 创建轨迹的线段几何对象
        traj_line = LineString(pts)
        
        # 计算轨迹与多边形障碍物的交集长度
        # 如果无交集，返回0；如果有交集，返回交集线段的总长度
        intersection = self.multi_poly.intersection(traj_line)
        
        # 获取交集的总长度
        if hasattr(intersection, 'length'):
            return intersection.length
        else:
            return 0.0


def create_cost_functions(q_min, q_max, env, qd, tolerance=1e-3):
    """
    创建各种代价函数
    
    参数:
    - q_min, q_max: 工作空间边界
    - env: 环境对象
    - qd: 目标位置（二维坐标）
    - tolerance: 目标到达容忍度
    
    返回:
    - 包含所有代价函数的字典
    """
    
    def loss_limits(candidates):
        """
        边界约束违反代价
        
        计算轨迹超出工作空间边界的违反程度
        """
        q = candidates['pos']  # 轨迹位置数据：(N_candidates, N_eval, ndof)
        
        # 计算下边界违反：当位置小于q_min时的违反量
        d_min = np.maximum(np.zeros_like(q), q_min - q)
        
        # 计算上边界违反：当位置大于q_max时的违反量  
        d_max = np.maximum(np.zeros_like(q), q - q_max)
        
        # 统计违反点的数量（每个候选轨迹的总违反点数）
        violations = (np.sum(d_min > 0.0, axis=(1,2)) + 
                     np.sum(d_max > 0.0, axis=(1,2)))
        
        return violations

    def loss_collision(candidates): 
        """
        碰撞代价
        
        计算每条候选轨迹与障碍物的碰撞严重程度
        """
        costs = []
        
        # 遍历每条候选轨迹
        for traj in candidates['pos']:
            # 计算轨迹与障碍物的交集长度
            collision_length = env.getTrajDist(traj)
            costs.append(collision_length)
        
        costs = np.array(costs)
        
        # 对任何碰撞（长度>0）添加额外惩罚
        # 这确保即使很小的碰撞也会被严重惩罚
        costs += (costs > 0.0)
        
        return costs

    def loss_curvature(candidates):
        """
        轨迹曲率代价
        
        计算轨迹的曲率，鼓励生成平滑的路径
        使用曲率公式：κ = |v × a| / |v|³
        """
        dq = candidates['vel']   # 速度：(N_candidates, N_eval, ndof)
        ddq = candidates['acc']  # 加速度：(N_candidates, N_eval, ndof)
        
        # 计算速度和加速度的模长平方
        dq_sq = np.sum(dq**2, axis=-1)      # |v|²
        ddq_sq = np.sum(ddq**2, axis=-1)    # |a|²
        
        # 计算速度和加速度的点积
        dq_ddq = np.sum(dq*ddq, axis=-1)    # v·a
        
        # 计算曲率平方：κ² = (|v|²|a|² - (v·a)²) / |v|⁶
        # 添加小的正则化项避免除零
        curvature_sq = (dq_sq * ddq_sq - dq_ddq**2) / (dq_sq**3 + 1e-6)
        
        # 返回平均曲率（沿轨迹时间维度）
        return np.mean(curvature_sq, axis=-1)

    def loss_target(candidates):
        """
        目标到达代价
        
        鼓励轨迹的最终位置接近目标位置
        """
        q = candidates['pos']
        
        # 计算最终位置与目标位置的欧式距离
        final_pos = q[:, -1, :]  # 取最终时刻的位置 (N_candidates, ndof)
        target_error = np.linalg.norm(final_pos - qd[np.newaxis, :], axis=1)
        
        # 对超出容忍度的误差添加额外惩罚
        costs = target_error + (target_error > tolerance)
        
        return costs
    
    def loss_combined(candidates):
        """
        组合代价函数
        
        将所有代价项按权重组合：
        - 主要目标：最小化时间
        - 约束：避免碰撞和边界违反
        - 次要目标：平滑性和目标到达精度
        """
        # 计算各项代价
        cost_curvature = loss_curvature(candidates)  # 平滑性代价
        cost_collision = loss_collision(candidates)  # 碰撞代价
        cost_limits = loss_limits(candidates)        # 边界违反代价
        cost_target = loss_target(candidates)        # 目标到达代价
        
        # 组合代价：
        # T: 时间（主要目标，权重=1）
        # 1e-3 * curvature: 平滑性（小权重，精细调节）
        # 1e3 * collision: 碰撞（大权重，硬约束）
        # 1e3 * limits: 边界（大权重，硬约束）
        # 1e2 * target: 目标到达（中等权重，软约束， 影响收敛）
        total_cost = (candidates['T'] + 
                     1e-3 * cost_curvature + 
                     1e3 * cost_collision + 
                     1e3 * cost_limits + 
                     1e3 * cost_target)
        
        return total_cost
    
    return {
        'limits': loss_limits,
        'collision': loss_collision, 
        'curvature': loss_curvature,
        'target': loss_target,
        'combined': loss_combined
    }


def setup_optimization_parameters():
    """
    设置VP-STO优化参数
    
    返回:
    - opt: 配置好的VPSTOOptions对象
    """
    # 创建2自由度的优化选项
    opt = VPSTOOptions(ndof=2)
    
    # 运动学约束
    opt.vel_lim = np.array([0.1, 0.1])    # 速度限制：[vx_max, vy_max]
    opt.acc_lim = np.array([0.5, 0.5])    # 加速度限制：[ax_max, ay_max]
    
    # 轨迹表示参数
    opt.N_via = 5        # 通径点数量：更多通径点=更灵活的轨迹
    opt.N_eval = 100     # 轨迹评估点数：用于代价函数计算
    
    # 优化算法参数
    opt.pop_size = 100    # CMA-ES种群大小：平衡探索能力和计算成本
    opt.max_iter = 200   # 最大迭代次数：充分优化
    opt.sigma_init = 5.5 # 初始标准差：较大的初始探索范围
    opt.log = True       # 启用日志记录以支持动画
    
    return opt


def run_trajectory_optimization(q0, q_min, q_max, env, qd, tolerance=1e-3):
    """
    运行轨迹优化
    
    参数:
    - q0: 起始位置
    - q_min, q_max: 工作空间边界
    - env: 环境对象
    - qd: 目标位置（二维坐标）
    - tolerance: 目标到达容忍度
    
    返回:
    - traj_opt: 配置好的VPSTO对象
    - sol: 优化解
    - cost_functions: 代价函数字典
    """
    print("Setting up trajectory optimization...")
    
    # 设置优化参数
    opt = setup_optimization_parameters()
    
    # 创建VP-STO优化器
    traj_opt = VPSTO(opt)
    
    # 创建代价函数
    cost_functions = create_cost_functions(q_min, q_max, env, qd, tolerance)
    
    print(f"Starting optimization from {q0} to target {qd}...")
    print(f"Workspace bounds: x∈[{q_min[0]}, {q_max[0]}], y∈[{q_min[1]}, {q_max[1]}]")
    print(f"Target tolerance: {tolerance}")
    
    # 运行优化
    # 注意：这里没有指定qT（最终位置），让算法自己优化最终位置
    sol = traj_opt.minimize(cost_functions['combined'], q0=q0)
    
    print(f"Optimization completed!")
    print(f"Best trajectory time: {sol.T_best:.4f} seconds")
    print(f"Final position: [{sol.w_best[-2]:.4f}, {sol.w_best[-1]:.4f}]")
    print(f"Target position: [{qd[0]:.4f}, {qd[1]:.4f}]")
    final_pos = np.array([sol.w_best[-2], sol.w_best[-1]])
    target_error = np.linalg.norm(final_pos - qd)
    print(f"Target error: {target_error:.6f}")
    
    return traj_opt, sol, cost_functions


def generate_trajectory_details(sol, n_points=1000):
    """
    生成详细的轨迹数据用于可视化和分析
    
    参数:
    - sol: 优化解
    - n_points: 轨迹插值点数
    
    返回:
    - t_traj: 时间序列
    - pos: 位置序列
    - vel: 速度序列  
    - acc: 加速度序列
    """
    # 生成时间序列
    t_traj = np.linspace(0, sol.T_best, n_points)
    
    # 计算位置、速度、加速度
    pos, vel, acc = sol.get_posvelacc(t_traj)
    
    return t_traj, pos, vel, acc


def plot_trajectory_results(q0, q_min, q_max, env, qd, pos, vel, acc, t_traj, save_figures=False):
    """
    绘制轨迹优化结果
    
    参数:
    - q0: 起始位置
    - q_min, q_max: 工作空间边界
    - env: 环境对象
    - qd: 目标位置（二维坐标）
    - pos, vel, acc: 轨迹数据
    - t_traj: 时间序列
    - save_figures: 是否保存图片
    
    返回:
    - 图形对象列表
    """
    figures = []
    
    # 1. 轨迹路径图
    plt.figure(figsize=(6,6))
    plt.xlim([q_min[0], q_max[0]])
    plt.ylim([q_min[1], q_max[1]])
    ax = plt.gca()
    
    # 绘制起始点
    plt.scatter(q0[0], q0[1], c='green', s=100, marker='o', 
               label='Start', zorder=5)
    
    # 绘制目标点
    plt.scatter(qd[0], qd[1], c='red', s=150, marker='*', 
               label=f'Target ({qd[0]:.3f}, {qd[1]:.3f})', zorder=5)
    
    # 绘制障碍物
    for i, pol in enumerate(env.poly_list):
        ax.add_patch(patches.Polygon(pol, facecolor='gray', alpha=0.7, 
                                   edgecolor='black', linewidth=1))
    
    # 绘制优化轨迹
    plt.plot(pos[:,0], pos[:,1], 'blue', linewidth=2, label='Optimized Path')
    
    # 标记最终位置
    plt.scatter(pos[-1,0], pos[-1,1], c='red', s=100, marker='x', 
               label='Final Position', zorder=5)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('2D Collision Avoidance - Optimized Final Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_figures:
        plt.savefig('trajectory_path.png', dpi=200, bbox_inches='tight')
    
    figures.append(plt.gcf())
    
    # 2. 速度剖面图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.grid(True, alpha=0.3)
    plt.title('Velocity Profile')
    plt.plot(t_traj, vel[:,0], label='Vx', linewidth=2)
    plt.plot(t_traj, vel[:,1], label='Vy', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.legend()
    
    # 3. 加速度剖面图
    plt.subplot(1, 2, 2)
    plt.grid(True, alpha=0.3)
    plt.title('Acceleration Profile')
    plt.plot(t_traj, acc[:,0], label='Ax', linewidth=2)
    plt.plot(t_traj, acc[:,1], label='Ay', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    
    plt.tight_layout()
    
    if save_figures:
        plt.savefig('velocity_acceleration_profiles.png', dpi=200, bbox_inches='tight')
        
    figures.append(plt.gcf())
    
    # 4. 速度大小和轨迹曲率
    plt.figure(figsize=(12, 4))
    
    # 速度大小
    plt.subplot(1, 2, 1)
    vel_magnitude = np.linalg.norm(vel, axis=1)
    plt.plot(t_traj, vel_magnitude, 'purple', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('Speed Profile')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    
    # 轨迹曲率估计
    plt.subplot(1, 2, 2)
    # 简单的曲率估计：使用相邻点的方向变化
    dx = np.diff(pos[:,0])
    dy = np.diff(pos[:,1])
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    
    # 避免除零错误
    ds = np.sqrt(dx[:-1]**2 + dy[:-1]**2) + 1e-8
    curvature = np.abs(dx[:-1]*ddy - dy[:-1]*ddx) / ds**3
    
    plt.plot(t_traj[:-2], curvature, 'orange', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.title('Trajectory Curvature')
    plt.xlabel('Time (s)')
    plt.ylabel('Curvature (1/m)')
    
    plt.tight_layout()
    
    if save_figures:
        plt.savefig('speed_curvature_analysis.png', dpi=200, bbox_inches='tight')
        
    figures.append(plt.gcf())
    
    return figures


def analyze_optimization_results(sol, cost_functions, env):
    """
    分析优化结果的详细信息
    
    参数:
    - sol: 优化解
    - cost_functions: 代价函数字典
    - env: 环境对象
    
    返回:
    - 分析结果字典
    """
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("="*50)
    
    # 获取最佳轨迹的详细数据
    t_traj = np.linspace(0, sol.T_best, 100)
    pos, vel, acc = sol.get_posvelacc(t_traj)
    
    # 构造候选轨迹格式用于代价函数分析
    best_candidate = {
        'pos': pos[np.newaxis, ...],  # 添加batch维度
        'vel': vel[np.newaxis, ...],
        'acc': acc[np.newaxis, ...],
        'T': np.array([sol.T_best])
    }
    
    # 计算各项代价
    cost_limits = cost_functions['limits'](best_candidate)[0]
    cost_collision = cost_functions['collision'](best_candidate)[0]
    cost_curvature = cost_functions['curvature'](best_candidate)[0]
    cost_target = cost_functions['target'](best_candidate)[0]
    cost_total = cost_functions['combined'](best_candidate)[0]
    
    # 计算轨迹统计信息
    max_speed = np.max(np.linalg.norm(vel, axis=1))
    max_acceleration = np.max(np.linalg.norm(acc, axis=1))
    total_distance = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    
    # 检查约束满足情况
    collision_length = env.getTrajDist(pos)
    is_collision_free = collision_length == 0.0
    
    print(f"Trajectory Duration: {sol.T_best:.4f} seconds")
    print(f"Total Distance: {total_distance:.4f} units")
    print(f"Average Speed: {total_distance/sol.T_best:.4f} units/s")
    print(f"Maximum Speed: {max_speed:.4f} units/s")
    print(f"Maximum Acceleration: {max_acceleration:.4f} units/s²")
    print(f"Final Position: [{pos[-1,0]:.4f}, {pos[-1,1]:.4f}]")
    print(f"Collision Free: {is_collision_free}")
    if not is_collision_free:
        print(f"  Collision Length: {collision_length:.6f}")
    
    print("\nCost Components:")
    print(f"  Time Cost: {sol.T_best:.4f}")
    print(f"  Curvature Cost: {cost_curvature:.6f}")
    print(f"  Collision Cost: {cost_collision:.6f}")
    print(f"  Limits Violation Cost: {cost_limits:.6f}")
    print(f"  Target Error Cost: {cost_target:.6f}")
    print(f"  Total Cost: {cost_total:.4f}")
    
    analysis_results = {
        'duration': sol.T_best,
        'distance': total_distance,
        'max_speed': max_speed,
        'max_acceleration': max_acceleration,
        'final_position': pos[-1],
        'is_collision_free': is_collision_free,
        'collision_length': collision_length,
        'costs': {
            'time': sol.T_best,
            'curvature': cost_curvature,
            'collision': cost_collision,
            'limits': cost_limits,
            'target': cost_target,
            'total': cost_total
        }
    }
    
    return analysis_results


def animate_vpsto_evolution(traj_opt, sol, env, q0, dq0=None, qT=None, dqT=None,
                             q_min=None, q_max=None, qd=None,
                             k_show=60, every=1, fps=20, save_path=None, figsize=(7,7)):
    """
    可视化 VP-STO 迭代过程（人群 -> 收敛）的动画。
    需要在优化前把 traj_opt.opt.log 设为 True。

    参数：
      traj_opt : VPSTO 实例（包含 vptraj）
      sol      : minimize 的返回（携带 candidates_list / loss_list 等日志）
      env      : 你的 CollisionEnvironment 实例（用于画多边形障碍）
      q0, dq0, qT, dqT : 与优化时相同的边界条件（dq0/dqT None 时按 0 处理）
      q_min, q_max : 工作空间边界，用于设定坐标轴
      qd       : 目标位置（二维坐标），用于绘制目标点
      k_show  : 每一帧显示的候选条数（按损失最小选前 k_show 条）
      every   : 隔多少代取一帧（降采样以加速/压缩）
      fps     : 动画帧率
      save_path : 若给定，如 'vpsto_evolution.gif' 则保存 GIF；否则只展示
      figsize : 画布大小
    """
    assert getattr(sol, 'candidates_list', None) is not None, \
        "需要在优化前设置 VPSTOOptions.log=True 才会有候选日志。"

    vptraj = traj_opt.vptraj
    ndof = traj_opt.opt.ndof
    if dq0 is None:
        dq0 = np.zeros(ndof)
    if dqT is None:
        dqT = np.zeros(ndof)

    # 画布 & 坐标轴
    fig, ax = plt.subplots(figsize=figsize)
    if q_min is not None and q_max is not None:
        ax.set_xlim([q_min[0], q_max[0]])
        ax.set_ylim([q_min[1], q_max[1]])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title(' evolution')

    # 画障碍物
    for pol in env.poly_list:
        ax.add_patch(patches.Polygon(pol, facecolor='gray', alpha=0.7,
                                     edgecolor='black', linewidth=1))

    # 绘制起始点
    ax.scatter(q0[0], q0[1], c='green', s=100, marker='o', 
               label='Start', zorder=5)

    # 绘制目标点
    if qd is not None:
        ax.scatter(qd[0], qd[1], c='red', s=150, marker='*', 
                  alpha=0.8, label=f'Target ({qd[0]:.3f}, {qd[1]:.3f})', zorder=5)

    # 颜色映射（损失小=更"好"的颜色）
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap('viridis')
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # 预先创建若干 Line2D 以复用（提升效率）
    # 注意：每个候选轨迹用一条线，mean用一条线，best用一条线
    pop_size = traj_opt.opt.pop_size
    lines = [ax.plot([], [], alpha=0.25, linewidth=1)[0] for _ in range(min(k_show, pop_size))]
    line_best, = ax.plot([], [], 'red', linewidth=2, alpha=1.0, label='best')
    line_mean, = ax.plot([], [], 'orange', linewidth=1, alpha=0.9, label='mean')
    ax.legend(loc='lower right')

    # 文本
    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

    # ---- 帧更新函数 ----
    iters = list(range(0, len(sol.candidates_list), every))

    def compute_trajs(p_batch):
        # 为这一代的整个人群计算轨迹（与优化时一致：先求最保守 T，再合成轨迹）
        # 处理自由终点的情况
        if qT is None:
            # 对于自由终点，p_batch包含通径点和终点位置
            # 从p_batch中提取终点信息（最后ndof个参数是终点位置）
            p_via_batch = p_batch[:, :-ndof]  # 通径点部分
            qT_batch = p_batch[:, -ndof:]     # 终点部分
            
            # 为每个候选轨迹计算时间
            T_batch = []
            pos_list = []
            for i in range(len(p_batch)):
                p_via = p_via_batch[i:i+1]  # 保持2D形状
                qT_single = qT_batch[i:i+1] # 保持2D形状
                T_single = vptraj.get_min_duration(p_via, q0[None, :], dq0[None, :], qT_single, dqT[None, :])[0]
                pos_single, _, _ = vptraj.get_trajectory(p_via, q0[None, :], dq0[None, :], qT_single, dqT[None, :], T_single)
                T_batch.append(T_single)
                pos_list.append(pos_single[0])
            
            pos = np.array(pos_list)
            T_batch = np.array(T_batch)
        else:
            T_batch = vptraj.get_min_duration(p_batch, q0, dq0, qT, dqT)
            pos, vel, acc = vptraj.get_trajectory(p_batch, q0, dq0, qT, dqT, T_batch)
        
        return pos, T_batch

    def compute_traj_single(p_vec):
        if qT is None:
            # 从p_vec中提取终点信息
            p_via_single = p_vec[:-ndof]
            qT_single = p_vec[-ndof:]
            T = vptraj.get_min_duration(p_via_single[None, :], q0[None, :], dq0[None, :], qT_single[None, :], dqT[None, :])[0]
            pos, _, _ = vptraj.get_trajectory(p_via_single[None, :], q0[None, :], dq0[None, :], qT_single[None, :], dqT[None, :], T)
            return pos[0], T
        else:
            T = vptraj.get_min_duration(p_vec[None, :], q0, dq0, qT, dqT)[0]
            pos, _, _ = vptraj.get_trajectory(p_vec[None, :], q0, dq0, qT, dqT, T)
            return pos[0], T

    cache = {}  # 简单缓存避免重复算
    def animate(frame_idx):
        i = iters[frame_idx]
        # 取本代数据
        p_batch = sol.candidates_list[i]
        costs   = sol.loss_list[i]
        p_mean  = sol.via_mean_list[i]
        p_best  = sol.via_best_list[i]

        # 计算/取缓存：这一代的人群轨迹、均值轨迹、最好轨迹
        if i not in cache:
            pos_batch, T_batch = compute_trajs(p_batch)
            pos_mean, _ = compute_traj_single(p_mean)  # 单条mean轨迹
            pos_best, _ = compute_traj_single(p_best)  # 单条best轨迹
            cache[i] = (pos_batch, costs, pos_mean, pos_best)
        else:
            pos_batch, costs, pos_mean, pos_best = cache[i]

        # 只显示损失最小的前 k_show 条候选轨迹
        order = np.argsort(costs)
        keep = order[:min(k_show, len(order))]
        cmin, cmax = costs[keep].min(), costs[keep].max()
        # 避免除零
        denom = (cmax - cmin) if (cmax > cmin) else 1.0
        for j, idx in enumerate(keep):
            q = pos_batch[idx]
            lines[j].set_data(q[:,0], q[:,1])
            # 归一化后上色（小损失 -> 较亮颜色）- 使用蓝色系避免与mean/best混淆
            color_val = 1.0 - (costs[idx] - cmin)/denom
            # 使用蓝色系颜色映射，避免与红色best和橙色mean混淆
            lines[j].set_color(plt.cm.Blues(0.3 + 0.7 * color_val))
            lines[j].set_alpha(0.3)
        # 其余隐藏
        for j in range(len(keep), len(lines)):
            lines[j].set_data([], [])

        # 画均值&最好轨迹 - 这里每个只画一条线
        # 确保mean和best在最上层显示
        line_mean.set_data(pos_mean[:,0], pos_mean[:,1])
        line_best.set_data(pos_best[:,0], pos_best[:,1])
        
        # 确保mean和best线条在候选轨迹之上
        line_mean.set_zorder(10)
        line_best.set_zorder(11)

        # 文本 - 添加更多调试信息
        txt.set_text(f'iter: {i+1}/{len(sol.candidates_list)}   '
                     f'best loss: {np.min(costs):.3g}\n'
                     f'showing {len(keep)} candidates + 1 mean + 1 best')

        return lines + [line_best, line_mean, txt]

    def init():
        for ln in lines:
            ln.set_data([], [])
        line_best.set_data([], [])
        line_mean.set_data([], [])
        txt.set_text('')
        return lines + [line_best, line_mean, txt]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(iters), interval=1000//fps,
                                   blit=False, repeat=False)

    if save_path is not None:
        try:
            anim.save(save_path, dpi=120, writer=animation.PillowWriter(fps=fps))
            print(f'Animation saved to {save_path}')
        except Exception as e:
            print('Save failed, showing interactively. Reason:', e)

    plt.tight_layout()
    plt.show()
    return anim


def main():    
    # 工作空间边界
    q_min = 0.0 * np.ones(2)    # 左下角：[0, 0]
    q_max = 0.5 * np.ones(2)    # 右上角：[0.5, 0.5]
    
    # 创建环境
    env = CollisionEnvironment()
    
    # 机器人起始位置
    q0 = np.array([0.15, 0.2])
    
    # 目标位置（二维坐标）
    qd = np.array([0.4, 0.3])
    tolerance = 1e-3  # 目标到达容忍度
    
    # 运行轨迹优化
    traj_opt, sol, cost_functions = run_trajectory_optimization(
        q0, q_min, q_max, env, qd, tolerance
    )
    
    # 生成详细轨迹数据
    t_traj, pos, vel, acc = generate_trajectory_details(sol, n_points=1000)

    # 重新启用动画功能
    SAVE_FLAG = False
    if SAVE_FLAG:
        save_path = f'evolution_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
    else:
        save_path = None
    anim = animate_vpsto_evolution(traj_opt, sol, env, q0,
                               q_min=q_min, q_max=q_max, qd=qd,
                               k_show=60, every=1, fps=20, save_path=save_path)

    
    # 分析优化结果
    analysis_results = analyze_optimization_results(sol, cost_functions, env)
    
    # 创建可视化
    figures = plot_trajectory_results(
        q0, q_min, q_max, env, qd, pos, vel, acc, t_traj, 
        save_figures=False  # 设为True可保存图片
    )
    
    # 显示图表
    plt.show()

    return {
        'optimizer': traj_opt,
        'solution': sol,
        'trajectory': (t_traj, pos, vel, acc),
        'analysis': analysis_results,
        'environment': env,
        'figures': figures
    }


if __name__ == "__main__":
    results = main()
    