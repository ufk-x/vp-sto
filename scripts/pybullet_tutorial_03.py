import time
import random
import numpy as np
import pybullet as p

from config import PANDA_URDF
from benchmarks.playground.env_simple_world import SimpleWorld
from benchmarks.utils import get_joint_limits  # 用于读取单个关节的上下限

def env_info_gen():
    robots_info = {
        "panda": {
            "urdf": PANDA_URDF,
            "base_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "conf": [0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4, 0.06, 0.06],
            "scale": 1,
        },
    }
    return {"robots_info": robots_info, "utilities": {}}

def clamp_angle(val, lower, upper):
    """将 val 限制在 [lower, upper]；若该关节为“环形”(上界<下界)，则不夹紧。"""
    if upper < lower:
        # utils.is_circular(...) 情况：无界旋转，不做 clamp
        return val
    return max(lower, min(upper, val))

def print_help():
    print(r"""
[Keyboard]
  1~7 : 对应第1~7个关节，每次按下让该关节随机转动一小步（弧度）
  8   : 张开夹爪
  9   : 闭合夹爪
  [ / ] : 减小 / 增大 随机步长(默认 5°)
  p   : 打印末端位姿 (xyz + rpy)
  r   : 重置机器人到初始姿态
  ESC : 退出
""")

if __name__ == "__main__":
    render = True
    mp4 = None

    # 创建环境 & 加载机器人
    env_infos = env_info_gen()
    env = SimpleWorld(use_gui=render, mp4=mp4)
    env.load_world(env_infos, robot=True)
    env.reset(env_infos)

    # 获取关节索引
    arm_joints = env.get_movable_joints(env.robot, gripper=False)  # 只要前7个臂关节 :contentReference[oaicite:2]{index=2}
    all_joints = env.get_movable_joints(env.robot, gripper=True)   # 包括夹爪两个关节   :contentReference[oaicite:3]{index=3}

    # 当前配置（含夹爪）
    q = list(env.get_joint_positions(env.robot))                   # 长度 = len(all_joints) :contentReference[oaicite:4]{index=4}
    assert len(q) == len(all_joints)
    N_ARM = len(arm_joints)  # 一般是7
    # 夹爪两个关节通常是 all_joints[-2], all_joints[-1]

    # 预取每个臂关节的限位，便于夹紧
    limits = []
    for j in arm_joints:
        low, up = get_joint_limits(env.robot, j)                   # 读取单关节上下限  :contentReference[oaicite:5]{index=5}
        limits.append((low, up))

    # 控制参数
    step_deg = 5.0  # 随机步长的上界（单位：度）
    step_rad_max = np.deg2rad(step_deg)

    print_help()
    print(f"[INFO] 初始随机步长：{step_deg:.1f}°")

    # 主循环：每帧处理键盘事件 + 步进物理
    while p.isConnected():
        # 处理键盘事件
        keys = p.getKeyboardEvents()
        if keys:
            # 退出
            if 27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED):  # ESC
                break

            # 打印末端位姿
            if ord('p') in keys and (keys[ord('p')] & p.KEY_WAS_TRIGGERED):
                eef_pose = env.get_eef_pose(env.robot)  # [x,y,z, roll,pitch,yaw]  :contentReference[oaicite:6]{index=6}
                print("[EEF] pos(xyz), rpy =", np.round(eef_pose, 4))

            # 重置
            if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
                env.reset(env_infos)
                q = list(env.get_joint_positions(env.robot))
                print("[INFO] 已重置到初始姿态。")
            
            # 调整步长
            if ord('[') in keys and (keys[ord('[')] & p.KEY_WAS_TRIGGERED):
                step_deg = max(0.5, step_deg - 1.0)
                step_rad_max = np.deg2rad(step_deg)
                print(f"[INFO] 随机步长 -> {step_deg:.1f}°")
            if ord(']') in keys and (keys[ord(']')] & p.KEY_WAS_TRIGGERED):
                step_deg = min(30.0, step_deg + 1.0)
                step_rad_max = np.deg2rad(step_deg)
                print(f"[INFO] 随机步长 -> {step_deg:.1f}°")

            # 夹爪控制（简单张开/闭合）
            if ord('8') in keys and (keys[ord('8')] & p.KEY_WAS_TRIGGERED):
                # 张开
                q[-2] = 0.06
                q[-1] = 0.06
                env.set_joint_positions(env.robot, q, gripper=True)  # 一次性下发全量配置  :contentReference[oaicite:7]{index=7}
                print("[GRIPPER] open -> 0.06")
            if ord('9') in keys and (keys[ord('9')] & p.KEY_WAS_TRIGGERED):
                # 闭合
                q[-2] = 0.02
                q[-1] = 0.02
                env.set_joint_positions(env.robot, q, gripper=True)
                print("[GRIPPER] close -> 0.02")

            # 数字键 1~7：对第 i 个臂关节做一次随机增量
            for i in range(N_ARM):  # i: 0..6
                key_code = ord(str(i+1))  # '1'..'7'
                if key_code in keys and (keys[key_code] & p.KEY_WAS_TRIGGERED):
                    delta = random.uniform(-step_rad_max, step_rad_max)
                    old_val = q[i]
                    low, up = limits[i]
                    new_val = clamp_angle(old_val + delta, low, up)
                    q[i] = new_val
                    env.set_joint_positions(env.robot, q, gripper=True)
                    print(f"[J{i+1}] {old_val:+.3f} -> {new_val:+.3f} rad (Δ={delta:+.3f}) | limits=({low:+.3f},{up:+.3f})")

        # 物理步进 & 限帧
        p.stepSimulation()
        time.sleep(env.dt)  # env.dt 在 SimpleWorld 里设置了 fixedTimeStep=1/100  :contentReference[oaicite:8]{index=8}
